"""
session_manager.py — Per-session state, interrupt logic, and task lifecycle.

Supports Telugu and Kannada language sessions.
Language is determined per-session via the WebSocket ?language= query param
(defaults to LANGUAGE env var, which defaults to "telugu").
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Dict, Optional

from config import config, get_language_config
from memory import ConversationMemory
from rag import RAGRetriever
from asr import ASRHandler, TranscriptResult
from tts import VoiceTTSHandler, TTSOrchestrator, schedule_tts_warmup
from llm import VoiceLLMClient

logger = logging.getLogger(__name__)

AudioSendCallback = Callable[[bytes], Awaitable[None]]
JsonSendCallback  = Callable[[dict], Awaitable[None]]


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """All state for a single connected client session."""
    session_id: str
    memory: ConversationMemory
    language: str
    audio_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    transcript_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    interrupt_event: asyncio.Event = field(default_factory=asyncio.Event)
    tts_cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    asr_task: Optional[asyncio.Task] = field(default=None, init=False)
    llm_tts_task: Optional[asyncio.Task] = field(default=None, init=False)
    greeted: bool = field(default=False, init=False)

    asr_handler: Optional[ASRHandler] = field(default=None, init=False)
    tts_handler: Optional[VoiceTTSHandler] = field(default=None, init=False)
    tts_orchestrator: Optional[TTSOrchestrator] = field(default=None, init=False)
    tts_orch_task: Optional[asyncio.Task] = field(default=None, init=False)
    llm_client: Optional[VoiceLLMClient] = field(default=None, init=False)
    bot_audio_active: bool = field(default=False, init=False)

    def cancel_tts(self) -> None:
        """Signal TTS to stop immediately."""
        self.tts_cancel_event.set()
        self.interrupt_event.set()
        self.bot_audio_active = False
        logger.info("TTS cancel signalled", extra={"session_id": self.session_id})

    async def cancel_and_wait_tts(self, timeout: float = 3.0) -> None:
        """
        Signal TTS to stop AND wait for the current orchestrator task to finish
        before clearing the cancel event.

        This prevents the race condition where:
          1. cancel_tts() sets the event
          2. reset clears it after 50 ms
          3. an MMS thread-executor still in-flight completes AFTER the reset
             and streams audio alongside the next response.
        """
        self.cancel_tts()
        task = self.tts_orch_task
        if task and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=0.3)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass
        self.interrupt_event.clear()
        self.tts_cancel_event.clear()
        logger.debug("TTS fully stopped and events reset",
                     extra={"session_id": self.session_id})

    def reset_for_new_turn(self) -> None:
        """Reset interrupt/cancel events for a new utterance (sync — no waiting)."""
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
        # Pre-load default language TTS model to eliminate cold-start latency
        schedule_tts_warmup(config.default_language)

    async def warmup_llm(self) -> None:
        """
        Send a tiny dummy request to Ollama so the 72B model is fully loaded
        into GPU VRAM before the first real user query arrives.

        Without this, the first inference triggers a ~100s cold-start load
        while the user is waiting — the model layers page in from disk and the
        session appears completely unresponsive.  With the warmup the model is
        already resident in VRAM and first-token latency drops to <1 s.
        """
        import httpx
        logger.info(
            "LLM warm-up starting — loading %s into VRAM (this takes ~60-120s on first boot)…",
            config.ollama.model,
        )
        try:
            async with httpx.AsyncClient(
                base_url=config.ollama.base_url,
                timeout=httpx.Timeout(300.0, connect=15.0),
            ) as client:
                t0 = asyncio.get_event_loop().time()
                resp = await client.post(
                    "/api/generate",
                    json={
                        "model": config.ollama.model,
                        "prompt": "hi",
                        "stream": False,
                        "options": {"num_predict": 1, "num_ctx": config.ollama.num_ctx},
                    },
                )
                resp.raise_for_status()
                elapsed = asyncio.get_event_loop().time() - t0
                logger.info(
                    "LLM warm-up complete — %s ready (%.1fs, first-token latency will now be <1s)",
                    config.ollama.model, elapsed,
                )
        except Exception as exc:
            logger.warning(
                "LLM warm-up failed (Ollama may not be running yet): %s", exc
            )

    def switch_asr_engine(self, session_id: str, engine: str) -> None:
        """Switch the STT engine for an active session without restarting it."""
        session = self._sessions.get(session_id)
        if session and session.asr_handler:
            session.asr_handler.set_engine(engine)
        else:
            logger.warning("switch_asr_engine: session %s not found", session_id)

    def create_session(
        self,
        send_audio_cb: AudioSendCallback,
        send_json_cb: JsonSendCallback,
        language: str = "telugu",
        voice: str = "male",
        tts_engine: str = "auto",
        stt_engine: str = "auto",
    ) -> Session:
        """
        Allocate a new session for the given language, wire all handlers,
        start async tasks.
        """
        # Validate and normalize language
        from config import SUPPORTED_LANGUAGES
        if language.lower() not in SUPPORTED_LANGUAGES:
            logger.warning(
                "Unsupported language '%s', defaulting to 'telugu'", language
            )
            language = "telugu"
        language = language.lower()

        lang_cfg = get_language_config(language)

        session_id = str(uuid.uuid4())
        memory = ConversationMemory(session_id=session_id)
        session = Session(session_id=session_id, memory=memory, language=language)

        # Wire ASR
        session.asr_handler = ASRHandler(
            session_id=session_id,
            audio_queue=session.audio_queue,
            transcript_queue=session.transcript_queue,
            interrupt_event=session.interrupt_event,
            language=language,
            stt_engine=stt_engine,
        )

        # Wire TTS
        session.tts_handler = VoiceTTSHandler(
            session_id=session_id,
            send_audio_cb=send_audio_cb,
            cancel_event=session.tts_cancel_event,
            language=language,
            voice=voice,
            tts_engine=tts_engine,
        )
        session.tts_orchestrator = TTSOrchestrator(
            session_id=session_id,
            tts_handler=session.tts_handler,
            cancel_event=session.tts_cancel_event,
        )

        # Wire LLM — RAG disabled for voice: encode() causes 4-5s page-fault
        # stall when weights are swapped out between turns. System prompt +
        # conversation history is sufficient context for voice interactions.
        session.llm_client = VoiceLLMClient(
            retriever=None,
            language=language,
        )

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
        logger.info(
            "Session created [%s / %s] stt=%s tts=%s",
            lang_cfg["display_name"],
            lang_cfg["display_name_native"],
            stt_engine,
            tts_engine,
            extra={"session_id": session_id},
        )
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

    def _drain_echo_transcripts(self, session: Session) -> None:
        """
        Discard any transcripts queued while the bot was speaking.

        When TTS audio plays through speakers, the microphone picks it up and
        ASR transcribes it as a new user utterance. Without draining, that echo
        transcript is picked up on the next loop iteration and interrupts or
        doubles the bot's response. We drain synchronously so the next
        ``await transcript_queue.get()`` always waits for genuine user speech.
        """
        drained = 0
        while not session.transcript_queue.empty():
            try:
                session.transcript_queue.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break
        if drained:
            logger.debug(
                "Echo suppression: discarded %d stale transcript(s) accumulated "
                "during TTS playback",
                drained,
                extra={"session_id": session.session_id},
            )

    async def _play_hardcoded(
        self,
        session: Session,
        send_json_cb: JsonSendCallback,
        text: str,
    ) -> None:
        """Synthesize and play a hardcoded string without the LLM.

        After all audio bytes are sent we wait for the client's playback
        duration to finish before returning.  Without this wait the server
        immediately starts the next synthesis while the client is still playing
        the previous message, causing audible overlap / interruption.
        """
        session.tts_handler.last_pcm_bytes_sent = 0   # reset byte counter
        session.tts_orchestrator = TTSOrchestrator(
            session_id=session.session_id,
            tts_handler=session.tts_handler,
            cancel_event=session.tts_cancel_event,
        )
        orch_task = asyncio.create_task(session.tts_orchestrator.run())
        session.tts_orch_task = orch_task
        await send_json_cb({"type": "tts_start"})
        await send_json_cb({"type": "bot_text_fragment", "text": text})
        await session.tts_orchestrator.fragment_queue.put(text)
        await session.tts_orchestrator.fragment_queue.put(None)
        try:
            await asyncio.wait_for(orch_task, timeout=30.0)
        except asyncio.TimeoutError:
            orch_task.cancel()
        session.bot_audio_active = False
        await send_json_cb({"type": "tts_end"})
        session.memory.add_bot_turn(text)

        # Wait briefly for ASR to transcribe any microphone echo from TTS playback,
        # then drain it. 400ms is enough for Sarvam STT round-trip without
        # blocking the loop for the full playback duration.
        await asyncio.sleep(0.4)

        # Drain any transcripts that arrived while the bot was speaking —
        # these are almost always microphone echo of the TTS output itself.
        self._drain_echo_transcripts(session)

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

        Phase 0 — Greeting (played proactively at session start, before any user input).
        Phase 1+ — Every user utterance goes to the LLM for a response.
        """
        lang_cfg = get_language_config(session.language)
        logger.info(
            "LLM/TTS loop started [%s]",
            lang_cfg["display_name"],
            extra={"session_id": session.session_id},
        )

        # Play greeting immediately at session start — before waiting for user input.
        # This ensures every user utterance always gets an LLM response.
        greeting = lang_cfg.get("greeting", "")
        if greeting:
            await self._play_hardcoded(session, send_json_cb, greeting)
        ivr_main_menu = lang_cfg.get("ivr_main_menu", "")
        if ivr_main_menu:
            await self._play_hardcoded(session, send_json_cb, ivr_main_menu)
        session.greeted = True

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

            # ASR engines may emit multiple final chunks for one long utterance
            # separated by very short pauses. Coalesce contiguous finals so the
            # LLM sees the full sentence and frontend transcript is complete.
            merged_parts = [user_text]
            # Adaptive merge window:
            # - If ASR final already ends a sentence, process immediately for
            #   lowest latency.
            # - Otherwise allow a short continuation window for split finals.
            first_word_count = len(user_text.split())
            if user_text and user_text[-1] in ".!?।":
                merge_timeout = 0.0
            else:
                merge_timeout = 0.20 if first_word_count <= 5 else 0.12
            while True:
                try:
                    more: TranscriptResult = await asyncio.wait_for(
                        session.transcript_queue.get(), timeout=merge_timeout
                    )
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    break
                if more.is_final and more.text.strip():
                    merged_parts.append(more.text.strip())
                    merge_timeout = 0.10
                # Ignore partial/empty here; they belong to ASR streaming state.
            user_text = " ".join(merged_parts).strip()

            logger.info(
                "Processing [%s]: %s",
                lang_cfg["display_name"],
                user_text[:80],
                extra={"session_id": session.session_id},
            )

            # Interrupt any ongoing TTS and wait until it fully stops
            # before clearing the cancel event — prevents old MMS thread
            # from streaming audio alongside the new response.
            await session.cancel_and_wait_tts(timeout=3.0)

            # Drain any transcripts that accumulated before this turn was
            # dequeued (user repeated themselves, Sarvam sent a duplicate,
            # or a prior slow turn caused a build-up).  Processing stale
            # transcripts would cancel the new response immediately after
            # it starts, making the bot appear to stop mid-sentence.
            _pre_llm_drained = 0
            while not session.transcript_queue.empty():
                try:
                    session.transcript_queue.get_nowait()
                    _pre_llm_drained += 1
                except asyncio.QueueEmpty:
                    break
            if _pre_llm_drained:
                logger.debug(
                    "Drained %d stale transcript(s) before LLM start",
                    _pre_llm_drained,
                    extra={"session_id": session.session_id},
                )

            session.memory.add_user_turn(user_text)
            await send_json_cb({"type": "transcript_final", "text": user_text})

            # ----------------------------------------------------------
            # LLM response for every user utterance
            # ----------------------------------------------------------
            await send_json_cb({"type": "tts_start"})

            # Clear any stale interrupt/cancel that arrived between
            # cancel_and_wait_tts clearing the events and now — prevents an
            # old barge-in signal from silently disabling this turn's TTS.
            session.reset_for_new_turn()

            session.tts_orchestrator = TTSOrchestrator(
                session_id=session.session_id,
                tts_handler=session.tts_handler,
                cancel_event=session.tts_cancel_event,
            )
            # Track this turn's audio output and detect "LLM text but no audio".
            bytes_before_turn = session.tts_handler.last_pcm_bytes_sent
            orch_task = asyncio.create_task(
                session.tts_orchestrator.run(),
                name=f"tts-orch-{session.session_id}",
            )
            session.tts_orch_task = orch_task

            full_bot_response = ""
            queued_fragment_this_turn = False
            try:
                async for fragment in session.llm_client.stream_response(
                    user_query=user_text,
                    memory=session.memory,
                    session_id=session.session_id,
                ):
                    # Always complete LLM generation even if TTS was cancelled
                    # (e.g. client VAD echo → "interrupt" message).  Stopping
                    # early stores a truncated response in memory, which
                    # confuses the next turn.  The TTS orchestrator stops
                    # on its own when cancel_event is set; we just stop
                    # feeding it new fragments.
                    full_bot_response += fragment + " "
                    await send_json_cb({"type": "bot_text_fragment", "text": fragment})
                    # Always allow the first fragment to enter TTS queue so a
                    # spurious early cancel cannot mute the entire reply.
                    if (not session.tts_cancel_event.is_set()) or (not queued_fragment_this_turn):
                        await session.tts_orchestrator.fragment_queue.put(fragment)
                        queued_fragment_this_turn = True

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

            bytes_after_turn = session.tts_handler.last_pcm_bytes_sent
            bytes_sent_this_turn = bytes_after_turn - bytes_before_turn
            if (
                bot_text
                and not session.tts_cancel_event.is_set()
                and bytes_sent_this_turn <= 0
            ):
                logger.warning(
                    "No TTS audio streamed for completed turn; retrying full response synthesis once",
                    extra={"session_id": session.session_id},
                )
                try:
                    await session.tts_handler.synthesize_and_stream(bot_text)
                except Exception as exc:
                    logger.error(
                        "TTS fallback synthesis failed: %s",
                        exc,
                        extra={"session_id": session.session_id},
                    )

            session.bot_audio_active = False
            await send_json_cb({"type": "tts_end"})
            logger.info(
                "Turn complete [%s]: %s",
                lang_cfg["display_name"],
                bot_text[:60],
                extra={"session_id": session.session_id},
            )

            # Drain transcripts that accumulated while the bot was speaking.
            # Prevents TTS audio echo from triggering a false barge-in on the
            # next loop iteration.
            self._drain_echo_transcripts(session)

        logger.info("LLM/TTS loop exiting", extra={"session_id": session.session_id})


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
session_manager = SessionManager()
