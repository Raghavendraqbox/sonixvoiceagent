"""
tts.py — Multi-language TTS handler for Dari and Pashto.

Priority chain per language:
  1. facebook/mms-tts-prs  (Dari  — Meta MMS VITS, local GPU)
     facebook/mms-tts-pbt  (Pashto — Meta MMS VITS, local GPU)
  2. edge-tts  fa-IR-DilaraNeural   (Dari  — Microsoft Azure, free)
               ps-AF-LatifaNeural   (Pashto — Microsoft Azure, free)
  3. gTTS  fa  (fallback for both languages)
  4. Silence padding  (last resort)

MMS-TTS models:
  - VITS architecture; natural, non-robotic pronunciation
  - Run fully on local GPU (CUDA), ~460 MB each
  - Output: 16 000 Hz PCM → resampled to 24 000 Hz for the browser
  - Dari/Pashto models require uroman romanisation: the VitsTokenizer
    handles this automatically when the `uroman` package is installed.

Audio output contract:
  PCM 16-bit signed LE, 24 000 Hz, mono
  Must match PLAYBACK_SAMPLE_RATE = 24000 in frontend/index.html

Cancel semantics:
  Setting cancel_event aborts synthesis between 60 ms chunks.
"""

import asyncio
import io
import logging
import threading
from typing import Callable, Awaitable

import numpy as np

from config import config, get_language_config

logger = logging.getLogger(__name__)

TTS_RATE = config.tts.sample_rate      # 24 000 Hz — browser playback rate

AudioSendCallback = Callable[[bytes], Awaitable[None]]

# ---------------------------------------------------------------------------
# MMS-TTS per-language model cache — loaded once per model_id
# ---------------------------------------------------------------------------
_mms_models: dict = {}          # {model_id: (model, tokenizer)} or {model_id: (None, None)}
_mms_lock = threading.Lock()


def _get_mms(model_id: str):
    """Return (model, tokenizer) for the given MMS model_id, loading on first call."""
    if model_id in _mms_models:
        return _mms_models[model_id]

    with _mms_lock:
        if model_id in _mms_models:
            return _mms_models[model_id]
        try:
            import torch
            from transformers import VitsModel, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading %s on %s …", model_id, device)

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = VitsModel.from_pretrained(model_id).to(device)
            model.eval()

            _mms_models[model_id] = (model, tokenizer)
            native_sr = getattr(model.config, "sampling_rate", 16_000)
            logger.info(
                "%s ready on %s (native %d Hz → output %d Hz)",
                model_id, device, native_sr, TTS_RATE,
            )
        except ImportError as exc:
            logger.error(
                "MMS-TTS load failed — missing dependency for %s: %s", model_id, exc
            )
            _mms_models[model_id] = (None, None)
        except Exception as exc:
            logger.error("MMS-TTS load failed for %s: %s", model_id, exc)
            _mms_models[model_id] = (None, None)

    return _mms_models[model_id]


def schedule_tts_warmup(language: str = "dari") -> None:
    """
    Pre-load the MMS-TTS model for the given language in a daemon thread.
    Call once at startup to eliminate cold-start latency on the first request.
    """
    lang_cfg = get_language_config(language)
    model_id = lang_cfg["mms_tts_model"]

    def _warmup():
        model, tokenizer = _get_mms(model_id)
        if model is None or tokenizer is None:
            return
        try:
            import torch
            # Synthesize a short phrase to compile CUDA kernels
            # Use a simple ASCII warmup string — uroman will romanise it safely
            warmup_text = "hello"
            inputs = tokenizer(warmup_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**inputs)
            logger.info("MMS-TTS warmup complete for %s (%s)", language, model_id)
        except Exception as exc:
            logger.warning("MMS-TTS warmup failed for %s: %s", model_id, exc)

    threading.Thread(target=_warmup, daemon=True, name=f"mms-warmup-{language}").start()


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _resample(audio: np.ndarray, from_hz: int, to_hz: int) -> np.ndarray:
    """
    Resample a float32 audio array.

    Priority:
      1. torchaudio.functional.resample  (best quality, sinc filter)
      2. scipy.signal.resample_poly       (good quality, polyphase)
      3. numpy linear interpolation       (always available)
    """
    if from_hz == to_hz:
        return audio

    import math
    g    = math.gcd(from_hz, to_hz)
    up   = to_hz   // g
    down = from_hz // g

    try:
        import torch, torchaudio  # type: ignore
        t = torch.from_numpy(audio).unsqueeze(0)
        r = torchaudio.functional.resample(t, from_hz, to_hz)
        return r.squeeze(0).numpy()
    except ImportError:
        pass

    try:
        from scipy.signal import resample_poly  # type: ignore
        return resample_poly(audio, up, down).astype(np.float32)
    except ImportError:
        pass

    n_out = int(len(audio) * to_hz / from_hz)
    x_old = np.linspace(0, 1, len(audio))
    x_new = np.linspace(0, 1, n_out)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _pcm_to_int16(audio_f32: np.ndarray) -> bytes:
    """Normalise float32 [-1,1] → int16 PCM bytes."""
    peak = np.max(np.abs(audio_f32))
    if peak > 0:
        audio_f32 = audio_f32 * (0.92 / peak)
    pcm = np.clip(audio_f32 * 32768.0, -32768, 32767).astype(np.int16)
    return pcm.tobytes()


def _apply_fade(audio: np.ndarray, rate: int, ms: int = 10) -> np.ndarray:
    """Apply short fade-in / fade-out to avoid click artefacts."""
    n = int(rate * ms / 1000)
    if len(audio) > n * 2:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        audio = audio.copy()
        audio[:n]  *= ramp
        audio[-n:] *= ramp[::-1]
    return audio


# ---------------------------------------------------------------------------
# VoiceTTSHandler
# ---------------------------------------------------------------------------

class VoiceTTSHandler:
    """
    Synthesizes Dari or Pashto text to PCM audio and streams it chunk-by-chunk.

    Primary  : facebook/mms-tts-prs (Dari) or facebook/mms-tts-pbt (Pashto)
               — local GPU VITS with automatic uroman romanisation
    Fallback1: edge-tts  fa-IR-DilaraNeural / ps-AF-LatifaNeural
    Fallback2: gTTS  fa (Pashto falls back to Persian for gTTS)
    Fallback3: Silence
    """

    def __init__(
        self,
        session_id: str,
        send_audio_cb: AudioSendCallback,
        cancel_event: asyncio.Event,
        language: str = "dari",
    ) -> None:
        self.session_id    = session_id
        self._send_audio   = send_audio_cb
        self._cancel_event = cancel_event

        lang_cfg = get_language_config(language)
        self._mms_model_id: str = lang_cfg["mms_tts_model"]
        self._mms_native_rate: int = lang_cfg["mms_tts_sample_rate"]
        self._edge_tts_voice: str = lang_cfg["edge_tts_voice"]
        self._gtts_language: str = lang_cfg["gtts_language"]
        self._language_display: str = lang_cfg["display_name"]
        # Once MMS fails once in a session, use edge-tts for all subsequent
        # sentences so the voice stays consistent (no mid-response voice switch).
        self._mms_available: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def synthesize_and_stream(self, text: str) -> bool:
        if not text.strip():
            return True
        logger.info(
            "TTS [%s]: %.70s",
            self._language_display,
            text,
            extra={"session_id": self.session_id},
        )
        return await self._synthesize_mms(text)

    # ------------------------------------------------------------------
    # MMS-TTS  (primary — VITS, local GPU)
    # ------------------------------------------------------------------

    async def _synthesize_mms(self, text: str) -> bool:
        """
        Synthesize with the language-specific MMS-TTS model on GPU.

        The VitsTokenizer automatically applies uroman romanisation for
        Dari/Pashto models (tokenizer.is_uroman == True) when the
        `uroman` package is installed. Runs inference in a thread pool
        to keep the event loop responsive.
        """
        if self._cancel_event.is_set():
            return False

        loop = asyncio.get_running_loop()

        def _infer() -> "np.ndarray | None":
            model, tokenizer = _get_mms(self._mms_model_id)
            if model is None or tokenizer is None:
                return None
            try:
                import torch
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    waveform = model(**inputs).waveform[0]
                return waveform.cpu().float().numpy()
            except ImportError as exc:
                logger.error(
                    "MMS-TTS uroman missing — install with: pip install uroman. Error: %s",
                    exc,
                    extra={"session_id": self.session_id},
                )
                return None
            except Exception as exc:
                logger.error(
                    "MMS-TTS inference error (%s): %s",
                    self._mms_model_id,
                    exc,
                    extra={"session_id": self.session_id},
                )
                return None

        try:
            audio_f32 = await loop.run_in_executor(None, _infer)
        except Exception as exc:
            logger.error("MMS-TTS executor error: %s", exc,
                         extra={"session_id": self.session_id})
            audio_f32 = None

        if audio_f32 is None:
            self._mms_available = False
            logger.warning(
                "MMS-TTS failed (%s) — switching to edge-tts for rest of session",
                self._mms_model_id,
                extra={"session_id": self.session_id},
            )
            return await self._synthesize_edge_tts(text)

        # Resample native rate → 24 kHz
        audio_f32 = _resample(audio_f32, self._mms_native_rate, TTS_RATE)
        audio_f32 = _apply_fade(audio_f32, TTS_RATE)
        pcm_bytes  = _pcm_to_int16(audio_f32)

        return await self._stream_pcm(pcm_bytes)

    # ------------------------------------------------------------------
    # edge-tts  (fallback 1)
    # ------------------------------------------------------------------

    async def _synthesize_edge_tts(self, text: str) -> bool:
        """Fallback to Microsoft edge-tts neural voice."""
        if self._cancel_event.is_set():
            return False

        try:
            import edge_tts
            import av
        except ImportError:
            return await self._synthesize_gtts(text)

        try:
            communicate = edge_tts.Communicate(text, voice=self._edge_tts_voice)
            audio_bytes = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes += chunk["data"]
                if self._cancel_event.is_set():
                    return False

            if not audio_bytes:
                return await self._synthesize_gtts(text)

            pcm_bytes = await asyncio.get_running_loop().run_in_executor(
                None, lambda: self._decode_mp3(audio_bytes)
            )
            if pcm_bytes is None:
                return await self._synthesize_gtts(text)

            return await self._stream_pcm(pcm_bytes)

        except Exception as exc:
            logger.error("edge-tts error: %s", exc,
                         extra={"session_id": self.session_id})
            return await self._synthesize_gtts(text)

    def _decode_mp3(self, audio_bytes: bytes) -> "bytes | None":
        """Decode MP3 → float32 PCM, resample to TTS_RATE, return int16 bytes."""
        try:
            import av
            buf       = io.BytesIO(audio_bytes)
            container = av.open(buf)
            resampler = av.audio.resampler.AudioResampler(
                format="s16", layout="mono", rate=TTS_RATE
            )
            frames: list = []
            for frame in container.decode(audio=0):
                for r in resampler.resample(frame):
                    frames.append(np.frombuffer(bytes(r.planes[0]), dtype=np.int16))
            for r in resampler.resample(None):
                frames.append(np.frombuffer(bytes(r.planes[0]), dtype=np.int16))
            container.close()

            if not frames:
                return None

            pcm = np.concatenate(frames)
            nz = np.where(np.abs(pcm) > 160)[0]
            if len(nz):
                pcm = pcm[nz[0]: nz[-1] + 1]

            f = pcm.astype(np.float32) / 32768.0
            f = _apply_fade(f, TTS_RATE)
            return _pcm_to_int16(f)

        except Exception as exc:
            logger.error("MP3 decode error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # gTTS  (fallback 2)
    # ------------------------------------------------------------------

    async def _synthesize_gtts(self, text: str) -> bool:
        if self._cancel_event.is_set():
            return False
        try:
            from gtts import gTTS
        except ImportError:
            return await self._synthesize_silence(text)

        try:
            loop = asyncio.get_running_loop()
            gtts_lang = self._gtts_language

            def _run():
                buf = io.BytesIO()
                gTTS(text=text, lang=gtts_lang, slow=False).write_to_fp(buf)
                return buf.getvalue()

            audio_bytes = await loop.run_in_executor(None, _run)
            if not audio_bytes or self._cancel_event.is_set():
                return not self._cancel_event.is_set()

            pcm_bytes = await loop.run_in_executor(
                None, lambda: self._decode_mp3(audio_bytes)
            )
            if pcm_bytes is None:
                return await self._synthesize_silence(text)

            return await self._stream_pcm(pcm_bytes)

        except Exception as exc:
            logger.error("gTTS error: %s", exc,
                         extra={"session_id": self.session_id})
            return await self._synthesize_silence(text)

    # ------------------------------------------------------------------
    # Silence  (last resort)
    # ------------------------------------------------------------------

    async def _synthesize_silence(self, text: str) -> bool:
        import struct
        spc     = TTS_RATE // 5
        silence = struct.pack(f"<{spc}h", *([0] * spc))
        for _ in text.split():
            if self._cancel_event.is_set():
                return False
            await self._send_audio(silence)
            await asyncio.sleep(0.2)
        return True

    # ------------------------------------------------------------------
    # Common PCM streamer
    # ------------------------------------------------------------------

    async def _stream_pcm(self, pcm_bytes: bytes) -> bool:
        """Send PCM bytes to the client in small chunks for low cancel latency."""
        bpc = int(TTS_RATE * config.tts.chunk_ms / 1000) * 2
        for i in range(0, len(pcm_bytes), bpc):
            if self._cancel_event.is_set():
                return False
            await self._send_audio(pcm_bytes[i: i + bpc])
            await asyncio.sleep(0)
        return not self._cancel_event.is_set()


# ---------------------------------------------------------------------------
# Backwards-compat alias
# ---------------------------------------------------------------------------
TeluguTTSHandler = VoiceTTSHandler


# ---------------------------------------------------------------------------
# TTSOrchestrator
# ---------------------------------------------------------------------------

class TTSOrchestrator:
    """Drains a queue of sentence fragments, synthesizes them in order."""

    def __init__(
        self,
        session_id: str,
        tts_handler: VoiceTTSHandler,
        cancel_event: asyncio.Event,
    ) -> None:
        self.session_id     = session_id
        self._tts           = tts_handler
        self._cancel_event  = cancel_event
        self._fragment_queue: asyncio.Queue = asyncio.Queue()
        self._active        = False

    @property
    def fragment_queue(self) -> asyncio.Queue:
        return self._fragment_queue

    async def run(self) -> None:
        self._active = True
        logger.debug("TTSOrchestrator started", extra={"session_id": self.session_id})

        while True:
            if self._cancel_event.is_set():
                while not self._fragment_queue.empty():
                    try:
                        self._fragment_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                break

            try:
                fragment = await asyncio.wait_for(self._fragment_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            if fragment is None:
                break

            if self._cancel_event.is_set():
                break

            completed = await self._tts.synthesize_and_stream(fragment)
            if not completed:
                while not self._fragment_queue.empty():
                    try:
                        self._fragment_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                break

        self._active = False
        logger.debug("TTSOrchestrator stopped", extra={"session_id": self.session_id})

    def is_active(self) -> bool:
        return self._active
