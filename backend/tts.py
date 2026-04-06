"""
tts.py — Native Telugu TTS handler.

Priority chain:
  1. facebook/mms-tts-tel  (Meta MMS VITS, native Telugu, local GPU)
  2. edge-tts te-IN-ShrutiNeural  (Microsoft neural, free, internet)
  3. gTTS  (Google TTS, free, internet)
  4. Silence padding  (last resort)

MMS-TTS (facebook/mms-tts-tel):
  - VITS architecture trained specifically on Telugu speech
  - Native, natural pronunciation — not robotic
  - Runs fully on local GPU (CUDA), ~460 MB model size
  - Output: 16 000 Hz PCM → resampled to 24 000 Hz for the browser

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

from config import config

logger = logging.getLogger(__name__)

TTS_RATE = config.tts.sample_rate  # 24 000 Hz — browser playback rate
MMS_NATIVE_RATE = 16_000           # facebook/mms-tts-tel native sample rate

AudioSendCallback = Callable[[bytes], Awaitable[None]]

# ---------------------------------------------------------------------------
# MMS-TTS singleton — loaded once on first synthesis call
# ---------------------------------------------------------------------------
_mms_model      = None
_mms_tokenizer  = None
_mms_lock       = threading.Lock()


def _get_mms():
    """Return (model, tokenizer), initialising on first call."""
    global _mms_model, _mms_tokenizer
    if _mms_model is not None:
        return _mms_model, _mms_tokenizer

    with _mms_lock:
        if _mms_model is not None:
            return _mms_model, _mms_tokenizer
        try:
            import torch
            from transformers import VitsModel, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading facebook/mms-tts-tel on %s …", device)
            tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-tel")
            model = VitsModel.from_pretrained("facebook/mms-tts-tel").to(device)
            model.eval()
            _mms_model = model
            _mms_tokenizer = tokenizer
            logger.info(
                "MMS-TTS Telugu ready on %s (native %d Hz → output %d Hz)",
                device, MMS_NATIVE_RATE, TTS_RATE,
            )
        except Exception as exc:
            logger.error("MMS-TTS load failed: %s", exc)

    return _mms_model, _mms_tokenizer


def schedule_tts_warmup():
    """Pre-load MMS-TTS model in a daemon thread at startup."""
    def _warmup():
        model, tokenizer = _get_mms()
        if model is None or tokenizer is None:
            return
        try:
            import torch
            # Synthesize a short Telugu phrase to compile CUDA kernels
            inputs = tokenizer("నమస్కారం", return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**inputs)
            logger.info("MMS-TTS warmup complete")
        except Exception as exc:
            logger.warning("MMS-TTS warmup failed: %s", exc)

    threading.Thread(target=_warmup, daemon=True, name="mms-warmup").start()


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _resample(audio: np.ndarray, from_hz: int, to_hz: int) -> np.ndarray:
    """
    Resample a float32 audio array using polyphase filtering.

    Priority:
      1. torchaudio.functional.resample  (best quality, sinc filter)
      2. scipy.signal.resample_poly       (good quality, polyphase)
      3. numpy linear interpolation       (always available, acceptable)
    """
    if from_hz == to_hz:
        return audio

    import math
    g    = math.gcd(from_hz, to_hz)
    up   = to_hz   // g
    down = from_hz // g

    # Option 1: torchaudio
    try:
        import torch, torchaudio  # type: ignore
        t = torch.from_numpy(audio).unsqueeze(0)
        r = torchaudio.functional.resample(t, from_hz, to_hz)
        return r.squeeze(0).numpy()
    except ImportError:
        pass

    # Option 2: scipy
    try:
        from scipy.signal import resample_poly  # type: ignore
        return resample_poly(audio, up, down).astype(np.float32)
    except ImportError:
        pass

    # Option 3: numpy linear interpolation
    n_out = int(len(audio) * to_hz / from_hz)
    x_old = np.linspace(0, 1, len(audio))
    x_new = np.linspace(0, 1, n_out)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _pcm_to_int16(audio_f32: np.ndarray) -> bytes:
    """Normalise float32 [-1,1] → int16 PCM bytes."""
    peak = np.max(np.abs(audio_f32))
    if peak > 0:
        audio_f32 = audio_f32 * (0.92 / peak)          # headroom below clipping
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
# TeluguTTSHandler
# ---------------------------------------------------------------------------

class TeluguTTSHandler:
    """
    Synthesizes Telugu text to PCM audio and streams it chunk-by-chunk.

    Primary  : facebook/mms-tts-tel  (local GPU VITS — native Telugu)
    Fallback1: edge-tts te-IN-ShrutiNeural  (Microsoft Azure, free)
    Fallback2: gTTS  (Google, free)
    Fallback3: Silence
    """

    def __init__(
        self,
        session_id: str,
        send_audio_cb: AudioSendCallback,
        cancel_event: asyncio.Event,
    ) -> None:
        self.session_id    = session_id
        self._send_audio   = send_audio_cb
        self._cancel_event = cancel_event

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def synthesize_and_stream(self, text: str) -> bool:
        if not text.strip():
            return True
        logger.info(
            "TTS: %.70s", text,
            extra={"session_id": self.session_id},
        )
        return await self._synthesize_mms(text)

    # ------------------------------------------------------------------
    # MMS-TTS  (primary — native Telugu VITS)
    # ------------------------------------------------------------------

    async def _synthesize_mms(self, text: str) -> bool:
        """
        Synthesize with facebook/mms-tts-tel on GPU.

        Runs inference in a thread pool executor so the event loop stays
        responsive.  Audio is resampled from 16 kHz → 24 kHz, then
        streamed in 60 ms chunks.
        """
        if self._cancel_event.is_set():
            return False

        loop = asyncio.get_running_loop()

        def _infer() -> np.ndarray | None:
            model, tokenizer = _get_mms()
            if model is None or tokenizer is None:
                return None
            try:
                import torch
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    waveform = model(**inputs).waveform[0]
                return waveform.cpu().float().numpy()
            except Exception as exc:
                logger.error(
                    "MMS-TTS inference error: %s", exc,
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
            logger.warning("MMS-TTS failed — falling back to edge-tts",
                           extra={"session_id": self.session_id})
            return await self._synthesize_edge_tts(text)

        # Resample 16 kHz → 24 kHz
        audio_f32 = _resample(audio_f32, MMS_NATIVE_RATE, TTS_RATE)
        audio_f32 = _apply_fade(audio_f32, TTS_RATE)
        pcm_bytes  = _pcm_to_int16(audio_f32)

        return await self._stream_pcm(pcm_bytes)

    # ------------------------------------------------------------------
    # edge-tts  (fallback 1)
    # ------------------------------------------------------------------

    async def _synthesize_edge_tts(self, text: str) -> bool:
        """Fallback to Microsoft edge-tts Telugu neural voice."""
        if self._cancel_event.is_set():
            return False

        try:
            import edge_tts
            import av
        except ImportError:
            return await self._synthesize_gtts(text)

        try:
            communicate  = edge_tts.Communicate(text, voice=config.tts.edge_tts_voice)
            audio_bytes  = b""
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

    def _decode_mp3(self, audio_bytes: bytes) -> bytes | None:
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
            # Trim silence boundaries
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

            def _run():
                buf = io.BytesIO()
                gTTS(text=text, lang=config.tts.gtts_language, slow=False).write_to_fp(buf)
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
        bpc = int(TTS_RATE * config.tts.chunk_ms / 1000) * 2   # bytes per chunk
        for i in range(0, len(pcm_bytes), bpc):
            if self._cancel_event.is_set():
                return False
            await self._send_audio(pcm_bytes[i: i + bpc])
            await asyncio.sleep(0)
        return not self._cancel_event.is_set()


# ---------------------------------------------------------------------------
# TTSOrchestrator
# ---------------------------------------------------------------------------

class TTSOrchestrator:
    """Drains a queue of sentence fragments, synthesizes them in order."""

    def __init__(
        self,
        session_id: str,
        tts_handler: TeluguTTSHandler,
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
                # Drain stale fragments before stopping
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
