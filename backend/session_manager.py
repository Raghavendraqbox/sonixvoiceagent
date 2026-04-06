"""
session_manager.py — Per-session state, interrupt logic, and task lifecycle.

Adapted from the English voice agent for Telugu:
  - Uses SonioxASRHandler instead of RivaASRHandler
  - Uses TeluguTTSHandler instead of RivaTTSHandler
  - Uses TeluguLLMClient instead of OllamaClient
  - Greeting is in Telugu
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Dict, Optional

from config import config
from memory import ConversationMemory
from rag import RAGRetriever
from asr import SonioxASRHandler, TranscriptResult
from tts import TeluguTTSHandler, TTSOrchestrator, schedule_tts_warmup
from llm import TeluguLLMClient

logger = logging.getLogger(__name__)

AudioSendCallback = Callable[[bytes], Awaitable[None]]
JsonSendCallback = Callable[[dict], Awaitable[None]]


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """All state for a single connected client session."""
    session_id: str
    memory: ConversationMemory
    audio_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    transcript_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    interrupt_event: asyncio.Event = field(default_factory=asyncio.Event)
    tts_cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    asr_task: Optional[asyncio.Task] = field(default=None, init=False)
    llm_tts_task: Optional[asyncio.Task] = field(default=None, init=False)
    greeted: bool = field(default=False, init=False)

    asr_handler: Optional[SonioxASRHandler] = field(default=None, init=False)
    tts_handler: Optional[TeluguTTSHandler] = field(default=None, init=False)
    tts_orchestrator: Optional[TTSOrchestrator] = field(default=None, init=False)
    llm_client: Optional[TeluguLLMClient] = field(default=None, init=False)

    def cancel_tts(self) -> None:
        """Signal TTS to stop immediately."""
        self.tts_cancel_event.set()
        self.interrupt_event.set()
        logger.info("TTS cancel signalled", extra={"session_id": self.session_id})

    def reset_for_new_turn(self) -> None:
        """Reset interrupt/cancel events for a new utterance."""
        self.interrupt_event.clear()
        self.tts_cancel_event.clear()
        logger.debug("Session events reset", extra={"session_id": self.session_id})

    async def cleanup(self) -> None:
        """Cancel all running tasks and close HTTP clients."""
        for task in (self.asr_task, self.llm_tts_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if self.asr_handler is not None:
            self.asr_handler.stop()
        if self.llm_client is not None:
            await self.llm_client.close()
        logger.info("Session cleanup complete", extra={"session_id": self.session_id})


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

class SessionManager:
    """Process-wide registry of active sessions."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self._retriever: Optional[RAGRetriever] = None

    def initialize_rag(self) -> None:
        """Build or reload the FAISS index at startup."""
        self._retriever = RAGRetriever()
        self._retriever.initialize()
        logger.info("RAG retriever initialized")
        schedule_tts_warmup()

    def create_session(
        self,
        send_audio_cb: AudioSendCallback,
        send_json_cb: JsonSendCallback,
    ) -> Session:
        """
        Allocate a new session, wire all handlers, start async tasks.
        """
        session_id = str(uuid.uuid4())
        memory = ConversationMemory(session_id=session_id)
        session = Session(session_id=session_id, memory=memory)

        # Wire ASR
        session.asr_handler = SonioxASRHandler(
            session_id=session_id,
            audio_queue=session.audio_queue,
            transcript_queue=session.transcript_queue,
            interrupt_event=session.interrupt_event,
        )

        # Wire TTS
        session.tts_handler = TeluguTTSHandler(
            session_id=session_id,
            send_audio_cb=send_audio_cb,
            cancel_event=session.tts_cancel_event,
        )
        session.tts_orchestrator = TTSOrchestrator(
            session_id=session_id,
            tts_handler=session.tts_handler,
            cancel_event=session.tts_cancel_event,
        )

        # Wire LLM
        session.llm_client = TeluguLLMClient(retriever=self._retriever)

        # Start background tasks
        session.asr_task = asyncio.create_task(
            session.asr_handler.run(),
            name=f"asr-{session_id}",
        )
        session.llm_tts_task = asyncio.create_task(
            self._llm_tts_loop(session, send_json_cb),
            name=f"llm_tts-{session_id}",
        )

        self._sessions[session_id] = session
        logger.info("Session created", extra={"session_id": session_id})
        return session

    async def destroy_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        await session.cleanup()

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _play_hardcoded(
        self,
        session: Session,
        send_json_cb: JsonSendCallback,
        text: str,
    ) -> None:
        """Synthesize and play a hardcoded string without the LLM."""
        session.tts_orchestrator = TTSOrchestrator(
            session_id=session.session_id,
            tts_handler=session.tts_handler,
            cancel_event=session.tts_cancel_event,
        )
        orch_task = asyncio.create_task(session.tts_orchestrator.run())
        await send_json_cb({"type": "tts_start"})
        await send_json_cb({"type": "bot_text_fragment", "text": text})
        await session.tts_orchestrator.fragment_queue.put(text)
        await session.tts_orchestrator.fragment_queue.put(None)
        try:
            await asyncio.wait_for(orch_task, timeout=30.0)
        except asyncio.TimeoutError:
            orch_task.cancel()
        await send_json_cb({"type": "tts_end"})
        session.memory.add_bot_turn(text)

    # ------------------------------------------------------------------
    # LLM + TTS conversation loop
    # ------------------------------------------------------------------

    async def _llm_tts_loop(
        self,
        session: Session,
        send_json_cb: JsonSendCallback,
    ) -> None:
        """
        Full conversation loop.

        Phase 0 — Greeting (deferred to first user utterance to avoid overlap):
            Play Telugu welcome message.
        Phase 1+ — Normal LLM turn:
            Stream LLM response → TTS.
        """
        logger.info("LLM/TTS loop started", extra={"session_id": session.session_id})

        while True:
            # Wait for a final transcript
            try:
                transcript: TranscriptResult = await session.transcript_queue.get()
            except asyncio.CancelledError:
                break

            if not transcript.is_final:
                continue

            user_text = transcript.text.strip()
            if not user_text:
                continue

            logger.info(
                "Processing: %s", user_text[:80],
                extra={"session_id": session.session_id},
            )

            # Interrupt any ongoing TTS
            session.cancel_tts()
            await asyncio.sleep(0.05)
            session.reset_for_new_turn()

            session.memory.add_user_turn(user_text)
            await send_json_cb({"type": "transcript_final", "text": user_text})

            # ----------------------------------------------------------
            # Phase 0: Telugu greeting (plays exactly once)
            # ----------------------------------------------------------
            if not session.greeted:
                session.greeted = True
                greeting = (
                    "Qobox కి స్వాగతం. నేను మీ వర్చువల్ అసిస్టెంట్‌ని. "
                    "నేను మీకు తెలుగు లేదా ఇంగ్లీష్‌లో సహాయం చేయగలను. "
                    "నేను మీకు ఎలా సహాయం చేయగలను?"
                )
                await self._play_hardcoded(session, send_json_cb, greeting)
                continue

            # ----------------------------------------------------------
            # Phase 1+: LLM response
            # ----------------------------------------------------------
            await send_json_cb({"type": "tts_start"})

            session.tts_orchestrator = TTSOrchestrator(
                session_id=session.session_id,
                tts_handler=session.tts_handler,
                cancel_event=session.tts_cancel_event,
            )
            orch_task = asyncio.create_task(
                session.tts_orchestrator.run(),
                name=f"tts-orch-{session.session_id}",
            )

            full_bot_response = ""
            try:
                async for fragment in session.llm_client.stream_response(
                    user_query=user_text,
                    memory=session.memory,
                    session_id=session.session_id,
                ):
                    if session.tts_cancel_event.is_set():
                        break
                    full_bot_response += fragment + " "
                    await send_json_cb({"type": "bot_text_fragment", "text": fragment})
                    await session.tts_orchestrator.fragment_queue.put(fragment)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(
                    "LLM error: %s", exc,
                    extra={"session_id": session.session_id},
                )
                await send_json_cb({"type": "error", "message": "LLM processing failed"})

            await session.tts_orchestrator.fragment_queue.put(None)

            try:
                await asyncio.wait_for(orch_task, timeout=30.0)
            except asyncio.TimeoutError:
                orch_task.cancel()
            except asyncio.CancelledError:
                orch_task.cancel()
                break

            bot_text = full_bot_response.strip()
            if bot_text:
                session.memory.add_bot_turn(bot_text)

            await send_json_cb({"type": "tts_end"})
            logger.info(
                "Turn complete: %s", bot_text[:60],
                extra={"session_id": session.session_id},
            )

        logger.info("LLM/TTS loop exiting", extra={"session_id": session.session_id})


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
session_manager = SessionManager()
