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
  1. Sarvam AI STT (SARVAM_API_KEY set)  ← primary, best for Indian languages
       Telugu  → language_code="te-IN"
       Kannada → language_code="kn-IN"
       Model: saarika:v2.5
  2. Soniox cloud ASR (SONIOX_API_KEY set)
       Telugu  → language_code="te"
       Kannada → language_code="kn"
  3. faster-whisper large-v3 on GPU
       Telugu  → language="te"
       Kannada → language="kn"
  4. Null stub                         ← placeholder only
"""

import asyncio
import io
import logging
import queue as _queue
import threading
import wave
from dataclasses import dataclass
from typing import Optional

import httpx

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

    stt_engine selects the ASR backend:
        "auto"    — Sarvam (if key) → Soniox (if key) → Whisper
        "sarvam"  — force Sarvam AI (falls back to Whisper if key missing)
        "soniox"  — force Soniox    (falls back to Whisper if key/SDK missing)
        "whisper" — force local Whisper large-v3
    """

    def __init__(
        self,
        session_id: str,
        audio_queue: asyncio.Queue,
        transcript_queue: asyncio.Queue,
        interrupt_event: asyncio.Event,
        language: str = "telugu",
        stt_engine: str = "auto",
    ) -> None:
        self.session_id = session_id
        self.audio_queue = audio_queue
        self.transcript_queue = transcript_queue
        self.interrupt_event = interrupt_event
        self._stopped = False
        self._force_restart = False  # set by set_engine() to switch mid-session
        self._soniox_failed = False  # set True on fatal Soniox errors → permanent Whisper fallback
        self._stt_engine: str = stt_engine.lower().strip()

        lang_cfg = get_language_config(language)
        self._soniox_language_code: str           = lang_cfg["soniox_language_code"]
        self._whisper_language: str               = lang_cfg["whisper_language"]
        self._sarvam_stt_language_code: str       = lang_cfg.get("sarvam_stt_language_code",        "te-IN")
        self._google_stt_language_code: str       = lang_cfg.get("google_stt_language_code",        "te-IN")
        self._azure_stt_language_code: str        = lang_cfg.get("azure_stt_language_code",         "te-IN")
        self._amazon_transcribe_language_code: str = lang_cfg.get("amazon_transcribe_language_code", "te-IN")
        self._language_display: str               = lang_cfg["display_name"]

    def set_engine(self, engine: str) -> None:
        """Switch STT engine mid-session. Takes effect after the current inner loop exits."""
        engine = engine.lower().strip()
        if engine not in ("auto", "sarvam", "soniox", "whisper", "google", "azure", "amazon"):
            logger.warning(
                "Unknown STT engine '%s' requested — ignoring", engine,
                extra={"session_id": self.session_id},
            )
            return
        logger.info(
            "STT engine switch requested: %s → %s",
            self._stt_engine, engine,
            extra={"session_id": self.session_id},
        )
        self._stt_engine = engine
        self._force_restart = True
        self._stopped = True  # breaks inner session loops that check self._stopped

    async def run(self) -> None:
        """Main ASR loop with exponential back-off reconnection.

        Engine selection (per self._stt_engine):
          "auto"    — Sarvam (if key) → Soniox (if key+SDK) → Whisper
          "sarvam"  — Sarvam AI only (Whisper fallback if key absent)
          "soniox"  — Soniox only    (Whisper fallback if unavailable)
          "whisper" — local Whisper large-v3 only

        Calling set_engine() sets _force_restart=True and _stopped=True.
        The outer loop sees _force_restart, resets both flags, and re-enters
        with the new engine — no task cancel needed.
        """
        backoff = 1.0
        while True:
            # Normal stop (session destroyed)
            if self._stopped and not self._force_restart:
                break
            # Engine switch requested — reset and re-enter with new engine
            if self._force_restart:
                self._stopped = False
                self._force_restart = False
                backoff = 1.0

            try:
                _sarvam_key = config.sarvam_stt.api_key
                _soniox_key = config.soniox.api_key

                _sarvam_available = bool(_sarvam_key) and not _sarvam_key.startswith("your-")
                _soniox_available = (
                    _SONIOX_AVAILABLE
                    and bool(_soniox_key)
                    and not _soniox_key.startswith("your-")
                    and not self._soniox_failed
                )

                engine = self._stt_engine

                if engine == "sarvam":
                    if _sarvam_available:
                        await self._run_sarvam_stt_session()
                    else:
                        logger.warning(
                            "Sarvam STT selected but SARVAM_API_KEY missing — falling back to Whisper",
                            extra={"session_id": self.session_id},
                        )
                        await self._run_whisper_session()
                elif engine == "soniox":
                    if _soniox_available:
                        await self._run_soniox_streaming()
                    else:
                        logger.warning(
                            "Soniox selected but unavailable (no key or SDK) — falling back to Whisper",
                            extra={"session_id": self.session_id},
                        )
                        await self._run_whisper_session()
                elif engine == "google":
                    if config.google_stt.api_key:
                        await self._run_google_stt_session()
                    else:
                        logger.warning(
                            "Google STT selected but GOOGLE_STT_API_KEY missing — falling back to Whisper",
                            extra={"session_id": self.session_id},
                        )
                        await self._run_whisper_session()
                elif engine == "azure":
                    if config.azure_stt.api_key:
                        await self._run_azure_stt_session()
                    else:
                        logger.warning(
                            "Azure STT selected but AZURE_STT_KEY missing — falling back to Whisper",
                            extra={"session_id": self.session_id},
                        )
                        await self._run_whisper_session()
                elif engine == "amazon":
                    if config.amazon_transcribe.access_key and config.amazon_transcribe.secret_key:
                        await self._run_amazon_transcribe_session()
                    else:
                        logger.warning(
                            "Amazon Transcribe selected but AWS credentials missing — falling back to Whisper",
                            extra={"session_id": self.session_id},
                        )
                        await self._run_whisper_session()
                elif engine == "whisper":
                    await self._run_whisper_session()
                else:  # "auto"
                    # Prefer streaming STT for full-duplex barge-in. Sarvam is
                    # accurate but batch-style, so it adds turn-taking latency.
                    if _soniox_available:
                        await self._run_soniox_streaming()
                    elif _sarvam_available:
                        await self._run_sarvam_stt_session()
                    else:
                        await self._run_whisper_session()

                backoff = 1.0
            except asyncio.CancelledError:
                break
            except _SonioxFatalError as exc:
                if self._stopped and not self._force_restart:
                    break
                self._soniox_failed = True
                logger.warning(
                    "Soniox fatal error (%s) — switching to Whisper large-v3 fallback",
                    exc,
                    extra={"session_id": self.session_id},
                )
                backoff = 1.0  # retry immediately with Whisper
            except Exception as exc:
                if self._stopped and not self._force_restart:
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
        self._force_restart = False

    # ------------------------------------------------------------------
    # Sarvam AI STT session (primary ASR)
    # ------------------------------------------------------------------

    async def _run_sarvam_stt_session(self) -> None:
        """
        VAD-based ASR using the Sarvam AI speech-to-text REST API.

        Same energy-based VAD loop as the Whisper fallback, but instead of
        running local model inference we POST the buffered utterance as WAV
        to https://api.sarvam.ai/speech-to-text (saarika:v2.5).

        This avoids loading Whisper large-v3 (~3 GB) into GPU VRAM, which
        would compete with qwen2.5:72b, and delivers higher accuracy for
        Telugu and Kannada than general-purpose ASR models.

        Audio format: PCM 16-bit signed LE, 16 kHz mono (same as Whisper).
        """
        try:
            import numpy as np
        except ImportError:
            logger.warning(
                "numpy not installed — cannot use Sarvam STT, falling back to Whisper",
                extra={"session_id": self.session_id},
            )
            await self._run_whisper_session()
            return

        logger.info(
            "Sarvam STT started (language=%s, model=%s)",
            self._sarvam_stt_language_code,
            config.sarvam_stt.model,
            extra={"session_id": self.session_id},
        )

        # VAD parameters — same as Whisper fallback
        SILENCE_RMS_THRESHOLD  = 0.008
        # 0.2s commit improves full-duplex responsiveness with Azure batch STT.
        SILENCE_FRAMES_TO_COMMIT = 2
        MIN_SPEECH_FRAMES       = 1
        MAX_SILENCE_FRAMES      = 30   # 3s hard reset

        audio_buf: list = []
        silence_frames   = 0
        speech_started   = False
        speech_frame_count = 0

        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
            while not self._stopped:
                try:
                    chunk: bytes = await asyncio.wait_for(
                        self.audio_queue.get(), timeout=0.5
                    )
                except asyncio.TimeoutError:
                    # Flush on timeout if we have a committed utterance
                    if (
                        speech_started
                        and silence_frames >= SILENCE_FRAMES_TO_COMMIT
                        and speech_frame_count >= MIN_SPEECH_FRAMES
                    ):
                        await self._sarvam_transcribe(client, audio_buf, np)
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
                                await self._sarvam_transcribe(client, audio_buf, np)
                            audio_buf = []
                            speech_started = False
                            silence_frames = 0
                            speech_frame_count = 0
                        elif silence_frames >= MAX_SILENCE_FRAMES:
                            audio_buf = []
                            speech_started = False
                            silence_frames = 0
                            speech_frame_count = 0

    async def _sarvam_transcribe(
        self, client: "httpx.AsyncClient", audio_buf: list, np
    ) -> None:
        """
        Build a WAV from the buffered PCM frames and POST to Sarvam STT.

        The Sarvam API accepts multipart/form-data with:
          file          — WAV audio bytes
          language_code — te-IN / kn-IN
          model         — saarika:v2.5
        """
        audio_array = np.concatenate(audio_buf).astype(np.float32)
        # Convert float32 [-1,1] back to int16 for WAV encoding
        pcm_int16 = (audio_array * 32767).clip(-32768, 32767).astype(np.int16)

        # Build WAV in memory (stdlib wave module)
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)           # 16-bit = 2 bytes
            wf.setframerate(16000)
            wf.writeframes(pcm_int16.tobytes())
        wav_bytes = wav_buf.getvalue()

        try:
            response = await client.post(
                config.sarvam_stt.endpoint,
                headers={"api-subscription-key": config.sarvam_stt.api_key},
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data={
                    "language_code": self._sarvam_stt_language_code,
                    "model": config.sarvam_stt.model,
                },
            )
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Sarvam STT HTTP error %d: %s",
                exc.response.status_code,
                exc.response.text[:200],
                extra={"session_id": self.session_id},
            )
            return
        except Exception as exc:
            logger.error(
                "Sarvam STT request failed: %s", exc,
                extra={"session_id": self.session_id},
            )
            return

        # Response schema: {"transcript": "...", ...}
        text = (result.get("transcript") or "").strip()
        if not text:
            return

        logger.info(
            "Sarvam STT [%s]: %s",
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
    # Google Cloud Speech-to-Text (VAD + REST batch)
    # Docs: https://cloud.google.com/speech-to-text/docs/reference/rest/v1/speech/recognize
    # ------------------------------------------------------------------

    async def _run_google_stt_session(self) -> None:
        try:
            import numpy as np
        except ImportError:
            logger.warning("numpy missing — Google STT unavailable, falling back to Whisper",
                           extra={"session_id": self.session_id})
            await self._run_whisper_session()
            return

        logger.info("Google STT started (language=%s)", self._google_stt_language_code,
                    extra={"session_id": self.session_id})

        SILENCE_RMS_THRESHOLD    = 0.008
        # 0.3s commit balances latency and long-utterance stability.
        SILENCE_FRAMES_TO_COMMIT = 3
        MIN_SPEECH_FRAMES        = 1
        MAX_SILENCE_FRAMES       = 30

        audio_buf: list = []
        silence_frames = 0
        speech_started = False
        speech_frame_count = 0

        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
            while not self._stopped:
                try:
                    chunk: bytes = await asyncio.wait_for(self.audio_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if (speech_started and silence_frames >= SILENCE_FRAMES_TO_COMMIT
                            and speech_frame_count >= MIN_SPEECH_FRAMES):
                        await self._google_transcribe(client, audio_buf, np)
                        audio_buf = []; speech_started = False
                        silence_frames = 0; speech_frame_count = 0
                    continue

                samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(samples ** 2)))

                if rms > SILENCE_RMS_THRESHOLD:
                    speech_started = True; silence_frames = 0
                    speech_frame_count += 1; audio_buf.append(samples)
                elif speech_started:
                    audio_buf.append(samples)
                    silence_frames += 1
                    if silence_frames >= SILENCE_FRAMES_TO_COMMIT:
                        if speech_frame_count >= MIN_SPEECH_FRAMES:
                            await self._google_transcribe(client, audio_buf, np)
                        audio_buf = []; speech_started = False
                        silence_frames = 0; speech_frame_count = 0
                    elif silence_frames >= MAX_SILENCE_FRAMES:
                        audio_buf = []; speech_started = False
                        silence_frames = 0; speech_frame_count = 0

    async def _google_transcribe(self, client: "httpx.AsyncClient", audio_buf: list, np) -> None:
        import base64
        audio_array = np.concatenate(audio_buf).astype(np.float32)
        pcm_int16 = (audio_array * 32767).clip(-32768, 32767).astype(np.int16)

        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2)
            wf.setframerate(16000); wf.writeframes(pcm_int16.tobytes())

        audio_b64 = base64.b64encode(wav_buf.getvalue()).decode("utf-8")
        url = f"https://speech.googleapis.com/v1/speech:recognize?key={config.google_stt.api_key}"
        payload = {
            "config": {
                "encoding": "LINEAR16",
                "sampleRateHertz": 16000,
                "languageCode": self._google_stt_language_code,
            },
            "audio": {"content": audio_b64},
        }
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if not results:
                return
            text = (results[0].get("alternatives", [{}])[0].get("transcript", "") or "").strip()
            if not text:
                return
            logger.info("Google STT [%s]: %s", self._language_display, text[:80],
                        extra={"session_id": self.session_id})
            if not self.interrupt_event.is_set():
                self.interrupt_event.set()
            await self.transcript_queue.put(TranscriptResult(text=text, is_final=True, confidence=1.0))
        except httpx.HTTPStatusError as exc:
            logger.error("Google STT HTTP error %d: %s", exc.response.status_code,
                         exc.response.text[:200], extra={"session_id": self.session_id})
        except Exception as exc:
            logger.error("Google STT error: %s", exc, extra={"session_id": self.session_id})

    # ------------------------------------------------------------------
    # Microsoft Azure Speech-to-Text (VAD + REST batch)
    # Docs: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/rest-speech-to-text
    # ------------------------------------------------------------------

    async def _run_azure_stt_session(self) -> None:
        try:
            import numpy as np
        except ImportError:
            logger.warning("numpy missing — Azure STT unavailable, falling back to Whisper",
                           extra={"session_id": self.session_id})
            await self._run_whisper_session()
            return

        region = config.azure_stt.region
        logger.info("Azure STT started (language=%s, region=%s)",
                    self._azure_stt_language_code, region,
                    extra={"session_id": self.session_id})

        SILENCE_RMS_THRESHOLD    = 0.008
        # 0.3s commit balances latency and long-utterance stability.
        SILENCE_FRAMES_TO_COMMIT = 3
        MIN_SPEECH_FRAMES        = 1
        MAX_SILENCE_FRAMES       = 30

        audio_buf: list = []
        silence_frames = 0
        speech_started = False
        speech_frame_count = 0

        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
            while not self._stopped:
                try:
                    chunk: bytes = await asyncio.wait_for(self.audio_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if (speech_started and silence_frames >= SILENCE_FRAMES_TO_COMMIT
                            and speech_frame_count >= MIN_SPEECH_FRAMES):
                        await self._azure_transcribe(client, audio_buf, np)
                        audio_buf = []; speech_started = False
                        silence_frames = 0; speech_frame_count = 0
                    continue

                samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(samples ** 2)))

                if rms > SILENCE_RMS_THRESHOLD:
                    speech_started = True; silence_frames = 0
                    speech_frame_count += 1; audio_buf.append(samples)
                elif speech_started:
                    audio_buf.append(samples)
                    silence_frames += 1
                    if silence_frames >= SILENCE_FRAMES_TO_COMMIT:
                        if speech_frame_count >= MIN_SPEECH_FRAMES:
                            await self._azure_transcribe(client, audio_buf, np)
                        audio_buf = []; speech_started = False
                        silence_frames = 0; speech_frame_count = 0
                    elif silence_frames >= MAX_SILENCE_FRAMES:
                        audio_buf = []; speech_started = False
                        silence_frames = 0; speech_frame_count = 0

    async def _azure_transcribe(self, client: "httpx.AsyncClient", audio_buf: list, np) -> None:
        audio_array = np.concatenate(audio_buf).astype(np.float32)
        pcm_int16 = (audio_array * 32767).clip(-32768, 32767).astype(np.int16)

        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2)
            wf.setframerate(16000); wf.writeframes(pcm_int16.tobytes())
        wav_bytes = wav_buf.getvalue()

        region = config.azure_stt.region
        url = (
            f"https://{region}.stt.speech.microsoft.com"
            f"/speech/recognition/conversation/cognitiveservices/v1"
            f"?language={self._azure_stt_language_code}&format=simple"
        )
        headers = {
            "Ocp-Apim-Subscription-Key": config.azure_stt.api_key,
            "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
            "Accept": "application/json",
        }
        try:
            resp = await client.post(url, headers=headers, content=wav_bytes)
            if resp.status_code == 401:
                logger.error("Azure STT: invalid key (401) — check AZURE_STT_KEY",
                             extra={"session_id": self.session_id})
                return
            resp.raise_for_status()
            data = resp.json()
            if data.get("RecognitionStatus") != "Success":
                return
            text = (data.get("DisplayText") or "").strip()
            if not text:
                return
            logger.info("Azure STT [%s]: %s", self._language_display, text[:80],
                        extra={"session_id": self.session_id})
            if not self.interrupt_event.is_set():
                self.interrupt_event.set()
            await self.transcript_queue.put(TranscriptResult(text=text, is_final=True, confidence=1.0))
        except httpx.HTTPStatusError as exc:
            logger.error("Azure STT HTTP error %d: %s", exc.response.status_code,
                         exc.response.text[:200], extra={"session_id": self.session_id})
        except Exception as exc:
            logger.error("Azure STT error: %s", exc, extra={"session_id": self.session_id})

    # ------------------------------------------------------------------
    # Amazon Transcribe Streaming (VAD + streaming SDK)
    # Requires: pip install amazon-transcribe
    # Docs: https://docs.aws.amazon.com/transcribe/latest/dg/streaming.html
    # ------------------------------------------------------------------

    async def _run_amazon_transcribe_session(self) -> None:
        try:
            from amazon_transcribe.client import TranscribeStreamingClient  # type: ignore
        except ImportError:
            logger.warning(
                "amazon-transcribe not installed — pip install amazon-transcribe. Falling back to Whisper.",
                extra={"session_id": self.session_id},
            )
            await self._run_whisper_session()
            return

        try:
            import numpy as np
        except ImportError:
            await self._run_whisper_session()
            return

        import os
        # Pass credentials via environment so TranscribeStreamingClient picks them up
        os.environ["AWS_ACCESS_KEY_ID"]     = config.amazon_transcribe.access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = config.amazon_transcribe.secret_key
        os.environ["AWS_DEFAULT_REGION"]    = config.amazon_transcribe.region

        logger.info("Amazon Transcribe started (language=%s, region=%s)",
                    self._amazon_transcribe_language_code, config.amazon_transcribe.region,
                    extra={"session_id": self.session_id})

        SILENCE_RMS_THRESHOLD    = 0.008
        # 0.3s commit balances latency and long-utterance stability.
        SILENCE_FRAMES_TO_COMMIT = 3
        MIN_SPEECH_FRAMES        = 1
        MAX_SILENCE_FRAMES       = 30

        audio_buf: list = []
        silence_frames = 0
        speech_started = False
        speech_frame_count = 0

        while not self._stopped:
            try:
                chunk: bytes = await asyncio.wait_for(self.audio_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if (speech_started and silence_frames >= SILENCE_FRAMES_TO_COMMIT
                        and speech_frame_count >= MIN_SPEECH_FRAMES):
                    await self._amazon_transcribe_utterance(audio_buf, np)
                    audio_buf = []; speech_started = False
                    silence_frames = 0; speech_frame_count = 0
                continue

            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(samples ** 2)))

            if rms > SILENCE_RMS_THRESHOLD:
                speech_started = True; silence_frames = 0
                speech_frame_count += 1; audio_buf.append(samples)
            elif speech_started:
                audio_buf.append(samples)
                silence_frames += 1
                if silence_frames >= SILENCE_FRAMES_TO_COMMIT:
                    if speech_frame_count >= MIN_SPEECH_FRAMES:
                        await self._amazon_transcribe_utterance(audio_buf, np)
                    audio_buf = []; speech_started = False
                    silence_frames = 0; speech_frame_count = 0
                elif silence_frames >= MAX_SILENCE_FRAMES:
                    audio_buf = []; speech_started = False
                    silence_frames = 0; speech_frame_count = 0

    async def _amazon_transcribe_utterance(self, audio_buf: list, np) -> None:
        from amazon_transcribe.client import TranscribeStreamingClient  # type: ignore
        from amazon_transcribe.handlers import TranscriptResultStreamHandler  # type: ignore
        from amazon_transcribe.model import TranscriptEvent  # type: ignore

        audio_array = np.concatenate(audio_buf).astype(np.float32)
        pcm_bytes = (audio_array * 32767).clip(-32768, 32767).astype(np.int16).tobytes()

        result_texts: list = []

        class _Handler(TranscriptResultStreamHandler):
            async def handle_transcript_event(self, event: TranscriptEvent):
                for result in event.transcript.results:
                    if not result.is_partial:
                        for alt in result.alternatives:
                            if alt.transcript:
                                result_texts.append(alt.transcript)

        try:
            client = TranscribeStreamingClient(region=config.amazon_transcribe.region)
            stream = await client.start_stream_transcription(
                language_code=self._amazon_transcribe_language_code,
                media_sample_rate_hz=16000,
                media_encoding="pcm",
            )
            handler = _Handler(stream.output_stream)

            CHUNK = 3200  # 100 ms at 16 kHz

            async def _send():
                async with stream.input_stream:
                    for i in range(0, len(pcm_bytes), CHUNK):
                        await stream.input_stream.send_audio_event(audio_chunk=pcm_bytes[i:i + CHUNK])
                        await asyncio.sleep(0)
                    await stream.input_stream.end_stream()

            await asyncio.gather(_send(), handler.handle_events())

            text = " ".join(result_texts).strip()
            if not text:
                return
            logger.info("Amazon Transcribe [%s]: %s", self._language_display, text[:80],
                        extra={"session_id": self.session_id})
            if not self.interrupt_event.is_set():
                self.interrupt_event.set()
            await self.transcript_queue.put(TranscriptResult(text=text, is_final=True, confidence=1.0))
        except Exception as exc:
            logger.error("Amazon Transcribe error: %s", exc, extra={"session_id": self.session_id})

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
        # 0.3s commit balances latency and long-utterance stability.
        SILENCE_FRAMES_TO_COMMIT = 3
        MIN_SPEECH_FRAMES = 1
        MAX_SILENCE_FRAMES = 30         # 3 s hard reset

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
