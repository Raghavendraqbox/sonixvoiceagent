"""
asr.py — Soniox streaming ASR handler for Telugu.

Reads raw PCM audio from an asyncio.Queue, streams it to the Soniox API,
and puts TranscriptResult objects onto an output queue.

Audio format contract (must match frontend):
  - Encoding:    PCM 16-bit signed little-endian (pcm_s16le)
  - Sample rate: 16 000 Hz
  - Channels:    1 (mono)
  - Chunk size:  ~3200 bytes (100 ms at 16kHz)

Priority chain:
  1. Soniox cloud ASR (SONIOX_API_KEY set)       ← best Telugu accuracy
  2. faster-whisper large-v3 on GPU              ← good Telugu, local
  3. Null stub                                   ← placeholder only
"""

import asyncio
import logging
import queue as _queue
import threading
from dataclasses import dataclass
from typing import Optional

from config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soniox SDK import guard
# ---------------------------------------------------------------------------
try:
    from soniox.transcribe_live import transcribe_stream
    from soniox.speech_service import SpeechClient
    _SONIOX_AVAILABLE = True
except ImportError:
    _SONIOX_AVAILABLE = False
    logger.warning(
        "soniox package not installed — ASR will fall back to Whisper large-v3. "
        "Install with: pip install soniox"
    )


# ---------------------------------------------------------------------------
# Transcript result
# ---------------------------------------------------------------------------

@dataclass
class TranscriptResult:
    """A single ASR result from the streaming recognizer."""
    text: str
    is_final: bool
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# SonioxASRHandler
# ---------------------------------------------------------------------------

class SonioxASRHandler:
    """
    Wraps the Soniox streaming ASR service for Telugu.

    Lifecycle:
        1. Construct once per session.
        2. Call `run()` as an asyncio task.
        3. Push raw PCM bytes into `audio_queue`.
        4. Consume TranscriptResult objects from `transcript_queue`.

    Falls back to faster-whisper large-v3 (GPU) when Soniox is unavailable.
    """

    def __init__(
        self,
        session_id: str,
        audio_queue: asyncio.Queue,
        transcript_queue: asyncio.Queue,
        interrupt_event: asyncio.Event,
    ) -> None:
        self.session_id = session_id
        self.audio_queue = audio_queue
        self.transcript_queue = transcript_queue
        self.interrupt_event = interrupt_event
        self._stopped = False

    async def run(self) -> None:
        """Main ASR loop with exponential back-off reconnection."""
        backoff = 1.0
        while not self._stopped:
            try:
                if _SONIOX_AVAILABLE and config.soniox.api_key:
                    await self._run_soniox_streaming()
                else:
                    await self._run_whisper_session()
                backoff = 1.0
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self._stopped:
                    break
                logger.error(
                    "ASR error (reconnecting in %.1fs): %s", backoff, exc,
                    extra={"session_id": self.session_id},
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def stop(self) -> None:
        """Signal the run loop to exit."""
        self._stopped = True

    # ------------------------------------------------------------------
    # Soniox streaming session
    # ------------------------------------------------------------------

    async def _run_soniox_streaming(self) -> None:
        """
        Stream audio to Soniox and forward transcript results.

        Runs the blocking Soniox SDK call in a daemon thread, bridged to the
        async loop via thread-safe queues.
        """
        loop = asyncio.get_running_loop()
        sync_audio_q: _queue.Queue = _queue.Queue()
        response_q: _queue.Queue = _queue.Queue()

        def audio_gen():
            """Synchronous generator consumed by the Soniox SDK."""
            while not self._stopped:
                try:
                    chunk = sync_audio_q.get(timeout=0.5)
                    if chunk is None:
                        return
                    yield chunk
                except _queue.Empty:
                    continue

        def soniox_thread():
            """Blocking Soniox call in a daemon thread."""
            try:
                with SpeechClient(api_key=config.soniox.api_key) as client:
                    for result in transcribe_stream(
                        audio_gen(),
                        client,
                        model=config.soniox.model,
                        language_code=config.soniox.language_code,
                        include_nonfinal=config.soniox.include_nonfinal,
                        audio_format=config.soniox.audio_format,
                        sample_rate_hertz=config.soniox.sample_rate_hertz,
                        num_audio_channels=config.soniox.num_audio_channels,
                    ):
                        response_q.put(result)
            except Exception as exc:
                response_q.put(exc)
            finally:
                response_q.put(None)  # sentinel

        thread = threading.Thread(target=soniox_thread, daemon=True, name="soniox-asr")
        thread.start()
        logger.info(
            "Soniox ASR started (language=%s, model=%s)",
            config.soniox.language_code,
            config.soniox.model,
            extra={"session_id": self.session_id},
        )

        async def pump_audio():
            """Pump async audio queue → sync queue for the Soniox thread."""
            while not self._stopped:
                try:
                    chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=1.0)
                    sync_audio_q.put(chunk)
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
            sync_audio_q.put(None)  # sentinel to unblock audio_gen

        pump_task = asyncio.create_task(pump_audio())

        try:
            while True:
                item = await loop.run_in_executor(None, response_q.get)
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                await self._process_soniox_result(item)
        finally:
            pump_task.cancel()
            try:
                await pump_task
            except asyncio.CancelledError:
                pass
            thread.join(timeout=2.0)

    async def _process_soniox_result(self, result) -> None:
        """
        Parse a Soniox result and push TranscriptResult to the output queue.

        Soniox results have a `tokens` list. Each token has `.text` and `.is_final`.
        A result is considered final when `result.final_proc_time_ms > 0` or all
        tokens are final (handles both v1 and v2 SDK formats).
        """
        if not hasattr(result, "tokens") or not result.tokens:
            return

        # Determine if this result is final
        is_final = False
        if hasattr(result, "final_proc_time_ms") and result.final_proc_time_ms > 0:
            is_final = True
        elif hasattr(result, "is_final"):
            is_final = result.is_final
        else:
            is_final = all(getattr(t, "is_final", False) for t in result.tokens)

        # Build the transcript text from tokens
        text = "".join(t.text for t in result.tokens).strip()
        if not text:
            return

        # Any speech activity interrupts TTS
        if not self.interrupt_event.is_set():
            self.interrupt_event.set()
            logger.debug(
                "Interrupt event set by ASR",
                extra={"session_id": self.session_id},
            )

        await self.transcript_queue.put(
            TranscriptResult(text=text, is_final=is_final, confidence=1.0)
        )
        logger.debug(
            "Soniox transcript %s: %s",
            "FINAL" if is_final else "partial",
            text[:60],
            extra={"session_id": self.session_id},
        )

    # ------------------------------------------------------------------
    # faster-whisper fallback (Telugu large-v3 on GPU)
    # ------------------------------------------------------------------

    async def _run_whisper_session(self) -> None:
        """
        Fallback ASR using faster-whisper large-v3 with Telugu language.

        Uses energy-based VAD to segment speech, then transcribes each utterance.
        Large-v3 gives the best accuracy for Telugu among open-source models.
        """
        try:
            import numpy as np
            from faster_whisper import WhisperModel
        except ImportError:
            logger.warning(
                "faster-whisper not installed — ASR in null stub mode",
                extra={"session_id": self.session_id},
            )
            await self._null_stub()
            return

        import torch
        if torch.cuda.is_available():
            device, compute = "cuda", "float16"
            logger.info("Whisper: GPU (CUDA)", extra={"session_id": self.session_id})
        else:
            device, compute = "cpu", "int8"
            logger.info("Whisper: CPU (slow for large-v3)", extra={"session_id": self.session_id})

        # Load model in executor to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        logger.info(
            "Loading Whisper large-v3 for Telugu (first run downloads ~3 GB)…",
            extra={"session_id": self.session_id},
        )
        model: "WhisperModel" = await loop.run_in_executor(
            None,
            lambda: WhisperModel("large-v3", device=device, compute_type=compute),
        )
        logger.info("Whisper large-v3 ready", extra={"session_id": self.session_id})

        # VAD parameters
        SILENCE_RMS_THRESHOLD = 0.008   # below → silence
        SILENCE_FRAMES_TO_COMMIT = 6    # 0.6 s silence ends utterance
        MIN_SPEECH_FRAMES = 1           # accept a single 100ms speech frame
        MAX_SILENCE_FRAMES = 20         # 2 s hard reset

        audio_buf: list = []
        silence_frames = 0
        speech_started = False
        speech_frame_count = 0

        while not self._stopped:
            try:
                chunk: bytes = await asyncio.wait_for(
                    self.audio_queue.get(), timeout=0.5
                )
            except asyncio.TimeoutError:
                if (
                    speech_started
                    and silence_frames >= SILENCE_FRAMES_TO_COMMIT
                    and speech_frame_count >= MIN_SPEECH_FRAMES
                ):
                    await self._whisper_transcribe(model, audio_buf, np)
                    audio_buf = []
                    speech_started = False
                    silence_frames = 0
                    speech_frame_count = 0
                continue

            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(samples ** 2)))

            if rms > SILENCE_RMS_THRESHOLD:
                speech_started = True
                silence_frames = 0
                speech_frame_count += 1
                audio_buf.append(samples)
            else:
                if speech_started:
                    audio_buf.append(samples)
                    silence_frames += 1
                    if silence_frames >= SILENCE_FRAMES_TO_COMMIT:
                        if speech_frame_count >= MIN_SPEECH_FRAMES:
                            await self._whisper_transcribe(model, audio_buf, np)
                        audio_buf = []
                        speech_started = False
                        silence_frames = 0
                        speech_frame_count = 0
                    elif silence_frames >= MAX_SILENCE_FRAMES:
                        audio_buf = []
                        speech_started = False
                        silence_frames = 0
                        speech_frame_count = 0

    async def _whisper_transcribe(self, model, audio_buf: list, np) -> None:
        """Transcribe buffered speech in a thread and emit the result."""
        audio_array = np.concatenate(audio_buf).astype(np.float32)
        loop = asyncio.get_running_loop()
        try:
            segments, info = await loop.run_in_executor(
                None,
                lambda: model.transcribe(
                    audio_array,
                    language="te",          # Telugu
                    beam_size=5,
                    vad_filter=True,        # built-in Whisper VAD
                    vad_parameters=dict(min_silence_duration_ms=500),
                ),
            )
            text = " ".join(seg.text for seg in segments).strip()
        except Exception as exc:
            logger.error(
                "Whisper transcription error: %s", exc,
                extra={"session_id": self.session_id},
            )
            return

        if not text:
            return

        logger.info(
            "Whisper transcript: %s", text[:80],
            extra={"session_id": self.session_id},
        )
        if not self.interrupt_event.is_set():
            self.interrupt_event.set()
        await self.transcript_queue.put(
            TranscriptResult(text=text, is_final=True, confidence=1.0)
        )

    # ------------------------------------------------------------------
    # Null stub
    # ------------------------------------------------------------------

    async def _null_stub(self) -> None:
        """Last-resort stub — drains audio silently."""
        logger.warning(
            "ASR in null stub mode — no transcripts will be generated",
            extra={"session_id": self.session_id},
        )
        while not self._stopped:
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            await asyncio.sleep(1.0)
