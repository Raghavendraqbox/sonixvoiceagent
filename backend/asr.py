"""
asr.py — Soniox streaming ASR handler for Telugu and Kannada.

Reads raw PCM audio from an asyncio.Queue, streams it to the Soniox API,
and puts TranscriptResult objects onto an output queue.

Audio format contract (must match frontend):
  - Encoding:    PCM 16-bit signed little-endian (pcm_s16le)
  - Sample rate: 16 000 Hz
  - Channels:    1 (mono)
  - Chunk size:  ~3200 bytes (100 ms at 16kHz)

Priority chain:
  1. Soniox cloud ASR (SONIOX_API_KEY set)
       Telugu  → language_code="te"
       Kannada → language_code="kn"
  2. faster-whisper large-v3 on GPU
       Telugu  → language="te"
       Kannada → language="kn"
  3. Null stub                         ← placeholder only
"""

import asyncio
import logging
import queue as _queue
import threading
from dataclasses import dataclass
from typing import Optional

from config import config, get_language_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal exceptions
# ---------------------------------------------------------------------------

class _SonioxFatalError(Exception):
    """Raised on non-retryable Soniox errors (e.g. 402 balance exhausted)."""


# ---------------------------------------------------------------------------
# Soniox SDK import guard (v2.x API)
# ---------------------------------------------------------------------------
try:
    from soniox import SonioxClient
    from soniox.realtime import RealtimeSTTConfig
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
# ASRHandler
# ---------------------------------------------------------------------------

class ASRHandler:
    """
    Wraps the Soniox streaming ASR service for Telugu and Kannada.

    Lifecycle:
        1. Construct once per session with desired language ("telugu" or "kannada").
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
        language: str = "telugu",
    ) -> None:
        self.session_id = session_id
        self.audio_queue = audio_queue
        self.transcript_queue = transcript_queue
        self.interrupt_event = interrupt_event
        self._stopped = False
        self._soniox_failed = False  # set True on fatal Soniox errors → permanent Whisper fallback

        lang_cfg = get_language_config(language)
        self._soniox_language_code: str = lang_cfg["soniox_language_code"]
        self._whisper_language: str = lang_cfg["whisper_language"]
        self._language_display: str = lang_cfg["display_name"]

    async def run(self) -> None:
        """Main ASR loop with exponential back-off reconnection."""
        backoff = 1.0
        while not self._stopped:
            try:
                use_soniox = _SONIOX_AVAILABLE and bool(config.soniox.api_key) and not self._soniox_failed
                if use_soniox:
                    await self._run_soniox_streaming()
                else:
                    await self._run_whisper_session()
                backoff = 1.0
            except asyncio.CancelledError:
                break
            except _SonioxFatalError as exc:
                if self._stopped:
                    break
                self._soniox_failed = True
                logger.warning(
                    "Soniox fatal error (%s) — switching to Whisper large-v3 fallback",
                    exc,
                    extra={"session_id": self.session_id},
                )
                backoff = 1.0  # retry immediately with Whisper
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
        Stream audio to Soniox v2 and forward transcript results.

        Uses two daemon threads (send + recv) bridged to the async loop via
        thread-safe queues, matching the v2 RealtimeSTTSession API.

        Root-cause fix: The Soniox SDK uses synchronous WebSocket I/O.
        Calling SonioxClient.__enter__() and connect() directly on the event-loop
        thread blocks it for the full TCP+TLS+WebSocket handshake (~200–800 ms).
        During that window the main WebSocket handler cannot read incoming audio
        frames from the browser, so the user's first words are silently dropped.
        Fix: run the blocking open/close calls in a thread-pool executor.
        """
        loop = asyncio.get_running_loop()
        sync_audio_q: _queue.Queue = _queue.Queue()
        response_q: _queue.Queue = _queue.Queue()

        soniox_cfg = RealtimeSTTConfig(
            model=config.soniox.model,
            audio_format=config.soniox.audio_format,
            sample_rate=config.soniox.sample_rate_hertz,
            num_channels=config.soniox.num_audio_channels,
            language_hints=[self._soniox_language_code],
            language_hints_strict=True,
            enable_language_identification=False,  # prevent auto-switching to English
        )

        # Open the Soniox WebSocket in a thread so the event loop stays free
        # to receive audio from the browser during the connection handshake.
        def _open_connection():
            client_obj = SonioxClient(api_key=config.soniox.api_key)
            client_obj.__enter__()
            sess = client_obj.realtime.stt.connect(config=soniox_cfg)
            sess.__enter__()
            return client_obj, sess

        client, session = await loop.run_in_executor(None, _open_connection)

        def recv_thread_fn():
            try:
                session.handle_events(lambda e: response_q.put(e))
            except Exception as exc:
                response_q.put(exc)
            finally:
                response_q.put(None)  # sentinel

        def send_thread_fn():
            while not self._stopped:
                try:
                    chunk = sync_audio_q.get(timeout=0.5)
                    if chunk is None:
                        break
                    session.send_byte_chunk(chunk)
                except _queue.Empty:
                    continue
                except Exception:
                    break

        recv_t = threading.Thread(target=recv_thread_fn, daemon=True, name="soniox-recv")
        send_t = threading.Thread(target=send_thread_fn, daemon=True, name="soniox-send")
        recv_t.start()
        send_t.start()

        logger.info(
            "Soniox ASR v2 started (language=%s [%s], model=%s)",
            self._soniox_language_code,
            self._language_display,
            config.soniox.model,
            extra={"session_id": self.session_id},
        )

        async def pump_audio():
            """Pump async audio queue → sync queue for the send thread."""
            while not self._stopped:
                try:
                    chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=1.0)
                    sync_audio_q.put(chunk)
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
            sync_audio_q.put(None)  # sentinel to unblock send thread

        pump_task = asyncio.create_task(pump_audio())

        try:
            while True:
                item = await loop.run_in_executor(None, response_q.get)
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                await self._process_soniox_event(item)
        finally:
            pump_task.cancel()
            try:
                await pump_task
            except asyncio.CancelledError:
                pass
            send_t.join(timeout=2.0)
            recv_t.join(timeout=2.0)
            # Close Soniox context managers in a thread (they make network calls)
            def _close_connection():
                try:
                    session.__exit__(None, None, None)
                except Exception:
                    pass
                try:
                    client.__exit__(None, None, None)
                except Exception:
                    pass
            await loop.run_in_executor(None, _close_connection)

    async def _process_soniox_event(self, event) -> None:
        """
        Parse a Soniox v2 RealtimeEvent and push TranscriptResult to the output queue.

        Events have a `tokens` list and `final_audio_proc_ms` (set when the batch is final).
        """
        if getattr(event, "error_code", None):
            raise _SonioxFatalError(f"{event.error_code}: {event.error_message}")

        if not event.tokens:
            return

        is_final = event.final_audio_proc_ms is not None
        text = "".join(t.text for t in event.tokens).strip()
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
    # faster-whisper fallback (GPU, large-v3)
    # ------------------------------------------------------------------

    async def _run_whisper_session(self) -> None:
        """
        Fallback ASR using faster-whisper large-v3 with the session language.

        Uses energy-based VAD to segment speech, then transcribes each utterance.
        Dari  → language="fa"  (Persian — best available for Afghan Dari)
        Pashto → language="ps"
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

        loop = asyncio.get_running_loop()
        logger.info(
            "Loading Whisper large-v3 for %s (first run downloads ~3 GB)…",
            self._language_display,
            extra={"session_id": self.session_id},
        )
        model: "WhisperModel" = await loop.run_in_executor(
            None,
            lambda: WhisperModel("large-v3", device=device, compute_type=compute),
        )
        logger.info(
            "Whisper large-v3 ready (%s / %s)",
            self._language_display,
            self._whisper_language,
            extra={"session_id": self.session_id},
        )

        # Drain audio that piled up while the model was loading (~2s).
        # Without this drain, all buffered chunks are processed at once →
        # multiple identical transcripts → multiple simultaneous TTS responses.
        drained = 0
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break
        if drained:
            logger.info(
                "Drained %d stale audio chunks accumulated during Whisper load",
                drained,
                extra={"session_id": self.session_id},
            )

        # VAD parameters
        SILENCE_RMS_THRESHOLD = 0.008
        SILENCE_FRAMES_TO_COMMIT = 6    # 0.6 s silence ends utterance
        MIN_SPEECH_FRAMES = 1
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
                    language=self._whisper_language,
                    beam_size=5,
                    vad_filter=True,
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
            "Whisper transcript [%s]: %s",
            self._language_display,
            text[:80],
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


# ---------------------------------------------------------------------------
# Backwards-compat alias
# ---------------------------------------------------------------------------
SonioxASRHandler = ASRHandler
