"""
tts.py — Multi-language TTS handler for Dari and Pashto.

Priority chain per language:
  Dari (strict):
    1. facebook/mms-tts-fas  (local GPU)
    2. Silence

  Pashto (configurable via PASHTO_TTS_ENGINE_PRIORITY):
    Engines available:
      mms        — facebook/mms-tts-pbt  (local GPU, default primary)
      elevenlabs — ElevenLabs REST API   (requires ELEVENLABS_API_KEY)
      narakeet   — Narakeet REST API     (requires NARAKEET_API_KEY)
      micmonster — MicMonster REST API   (requires MICMONSTER_API_KEY)
      speakatoo  — Speakatoo REST API    (requires SPEAKATOO_API_KEY)
      edge       — Microsoft edge-tts    (free, ps-AF-LatifaNeural / GulNawazNeural)
      gtts       — Google gTTS           (fallback, uses Persian fa)

    Default priority: mms,edge,gtts
    Override via env: PASHTO_TTS_ENGINE_PRIORITY=elevenlabs,edge,gtts

Audio output contract:
  PCM 16-bit signed LE, 24 000 Hz, mono
  Must match PLAYBACK_SAMPLE_RATE = 24000 in frontend/index.html

Cancel semantics:
  Setting cancel_event aborts synthesis between 60 ms chunks.
"""

import asyncio
import datetime
import io
import logging
import os
import threading
import wave
from pathlib import Path
from typing import Callable, Awaitable

import numpy as np

from config import config, get_language_config

logger = logging.getLogger(__name__)

TTS_RATE = config.tts.sample_rate      # 24 000 Hz — browser playback rate

AudioSendCallback = Callable[[bytes], Awaitable[None]]


def _debug_dump_audio_pair(
    provider: str,
    session_id: str,
    source_audio: bytes,
    source_ext: str,
    decoded_pcm: bytes,
) -> None:
    """
    Optionally dump source+decoded audio to disk for A/B debugging.
    Disabled unless DEBUG_TTS_DUMP_AUDIO=true.
    """
    if not config.tts.debug_dump_audio:
        return
    try:
        out_dir = Path(config.tts.debug_dump_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
        sid = "".join(ch for ch in session_id if ch.isalnum() or ch in ("-", "_"))[:32] or "session"
        stem = f"{provider}-{sid}-{timestamp}"

        source_path = out_dir / f"{stem}.{source_ext.lstrip('.')}"
        wav_path = out_dir / f"{stem}.decoded.wav"

        source_path.write_bytes(source_audio)

        # Save decoded PCM as a standard WAV for direct playback/comparison.
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)   # int16
            wf.setframerate(TTS_RATE)
            wf.writeframes(decoded_pcm)

        logger.info(
            "Debug audio dump saved: %s and %s",
            source_path,
            wav_path,
            extra={"session_id": session_id},
        )
    except Exception as exc:
        logger.warning(
            "Debug audio dump failed: %s",
            exc,
            extra={"session_id": session_id},
        )

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


def _bandpass_filter(audio: np.ndarray, rate: int,
                     low_hz: float = 80.0, high_hz: float = 8000.0) -> np.ndarray:
    """
    Butterworth bandpass filter — keeps human speech frequencies (80 Hz–8 kHz)
    and removes sub-bass rumble, ultrasonic hiss, and encoder artefacts.
    """
    try:
        from scipy.signal import butter, sosfilt
        nyq    = rate / 2.0
        lo     = max(low_hz  / nyq, 1e-4)
        hi     = min(high_hz / nyq, 0.9999)
        sos    = butter(4, [lo, hi], btype="bandpass", output="sos")
        return sosfilt(sos, audio).astype(np.float32)
    except Exception:
        return audio  # scipy unavailable — skip filter


def _smooth_noise_gate(audio: np.ndarray, rate: int,
                       threshold: float = 0.012,
                       window_ms: int = 10) -> np.ndarray:
    """
    Smooth RMS-based noise gate — avoids click artefacts from hard clipping.

    Computes RMS over overlapping windows; windows whose RMS falls below
    `threshold` are faded to silence using a cosine taper so there are no
    discontinuities at the gate boundary.
    """
    win   = max(1, int(rate * window_ms / 1000))
    out   = audio.copy()
    n     = len(audio)
    for start in range(0, n, win):
        end  = min(start + win, n)
        seg  = audio[start:end]
        rms  = float(np.sqrt(np.mean(seg ** 2)))
        if rms < threshold:
            # cosine fade to zero so boundary is smooth
            taper = np.cos(np.linspace(0, np.pi / 2, end - start)).astype(np.float32)
            out[start:end] = seg * taper * (rms / threshold if threshold > 0 else 0)
    return out


def _mp3_bytes_to_pcm(audio_bytes: bytes, denoise: bool = False) -> "bytes | None":
    """
    Decode MP3 bytes → int16 PCM at TTS_RATE using PyAV.

    Args:
        denoise: Apply bandpass + noise gate. Set True for local MMS output
                 which has a noisy floor. Leave False for cloud TTS (ElevenLabs,
                 Narakeet, etc.) which already produce clean audio — noise
                 reduction would gate out quiet consonants and degrade quality.

    Pipeline:
      1. (if denoise) Bandpass 80 Hz–8 kHz  — removes sub-bass rumble & hiss
      2. (if denoise) Smooth RMS noise gate  — fades quiet windows (no clicks)
      3. Silence trim           — clean start/end cuts
      4. Fade in/out            — avoids click artefacts at boundaries
      5. Level cap at 0.92      — only reduce if near clipping; never boost
    """
    try:
        import av
        buf       = io.BytesIO(audio_bytes)
        container = av.open(buf)
        resampler = av.audio.resampler.AudioResampler(
            format="s16", layout="mono", rate=TTS_RATE
        )
        frames: list[np.ndarray] = []
        for frame in container.decode(audio=0):
            for r in resampler.resample(frame):
                # Use ndarray extraction instead of raw plane bytes.
                # Raw plane buffers may include padding/stride bytes and can
                # introduce audible static when interpreted as contiguous PCM.
                arr = r.to_ndarray()
                if arr is None:
                    continue
                if arr.ndim == 2:
                    # Expected shape for mono is (1, samples).
                    # If channel axis differs, flatten safely to 1-D PCM.
                    if arr.shape[0] == 1:
                        arr = arr[0]
                    elif arr.shape[1] == 1:
                        arr = arr[:, 0]
                    else:
                        arr = arr.reshape(-1)
                frames.append(arr.astype(np.int16, copy=False))
        for r in resampler.resample(None):
            arr = r.to_ndarray()
            if arr is None:
                continue
            if arr.ndim == 2:
                if arr.shape[0] == 1:
                    arr = arr[0]
                elif arr.shape[1] == 1:
                    arr = arr[:, 0]
                else:
                    arr = arr.reshape(-1)
            frames.append(arr.astype(np.int16, copy=False))
        container.close()

        if not frames:
            return None

        pcm = np.concatenate(frames)
        f   = pcm.astype(np.float32) / 32768.0

        if denoise:
            # Step 1 — bandpass: keep only speech-range frequencies
            f = _bandpass_filter(f, TTS_RATE)

            # Step 2 — smooth noise gate (no clicks)
            f = _smooth_noise_gate(f, TTS_RATE, threshold=0.012, window_ms=10)

        # Step 3 — trim edges (silence + noise-only start/end)
        nz = np.where(np.abs(f) > 0.005)[0]
        if len(nz) == 0:
            return None
        f = f[nz[0]: nz[-1] + 1]

        # Step 4 — fade + level cap (never boost)
        f    = _apply_fade(f, TTS_RATE)
        peak = float(np.max(np.abs(f))) if len(f) else 0.0
        if peak < 1e-6:
            return None
        if peak > 0.92:
            f = f * (0.92 / peak)

        pcm_out = np.clip(f * 32768.0, -32768, 32767).astype(np.int16)
        return pcm_out.tobytes()

    except Exception as exc:
        logger.error("MP3 decode error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# VoiceTTSHandler
# ---------------------------------------------------------------------------

class VoiceTTSHandler:
    """
    Synthesizes Dari or Pashto text to PCM audio and streams it chunk-by-chunk.

    Dari (strict):
      Primary  : facebook/mms-tts-fas (local GPU)
      Fallback : Silence

    Pashto (order set by PASHTO_TTS_ENGINE_PRIORITY):
      mms        — facebook/mms-tts-pbt (local GPU)
      elevenlabs — ElevenLabs REST API
      narakeet   — Narakeet REST API
      micmonster — MicMonster REST API
      speakatoo  — Speakatoo REST API
      edge       — Microsoft edge-tts (ps-AF)
      gtts       — Google gTTS (Persian fallback)
    """

    def __init__(
        self,
        session_id: str,
        send_audio_cb: AudioSendCallback,
        cancel_event: asyncio.Event,
        language: str = "dari",
        voice: str = "male",
        tts_engine: str = "auto",
    ) -> None:
        self.session_id    = session_id
        self._send_audio   = send_audio_cb
        self._cancel_event = cancel_event
        self._language     = language.lower()
        self._voice        = voice.lower()
        self.last_pcm_bytes_sent: int = 0  # tracks bytes sent in last synthesis

        lang_cfg = get_language_config(language)
        self._mms_model_id: str     = lang_cfg["mms_tts_model"]
        self._mms_native_rate: int  = lang_cfg["mms_tts_sample_rate"]
        self._gtts_language: str    = lang_cfg["gtts_language"]
        self._language_display: str = lang_cfg["display_name"]
        self._lang_cfg              = lang_cfg

        # Dari must remain strict Dari-only (MMS model). Keep voice metadata for
        # Pashto/cloud providers, but never switch Dari to non-Dari engines.
        if voice == "female":
            self._edge_tts_voice: str       = lang_cfg["edge_tts_voice"]
            self._use_edge_primary: bool    = False
        else:
            self._edge_tts_voice: str       = lang_cfg.get("edge_tts_voice_male", lang_cfg["edge_tts_voice"])
            self._use_edge_primary: bool    = False

        # Build Pashto engine priority list
        # If a specific engine is requested from the UI, put it first with fallbacks.
        # "auto" → use PASHTO_TTS_ENGINE_PRIORITY from .env
        if self._language == "pashto":
            VALID_ENGINES = {"mms", "elevenlabs", "narakeet", "micmonster", "speakatoo", "edge", "gtts"}
            if tts_engine and tts_engine != "auto" and tts_engine in VALID_ENGINES:
                # Place the chosen engine first with appropriate fallbacks.
                # Cloud engines (elevenlabs, narakeet, etc.) fall back only to other
                # cloud/neural engines — never MMS, which sounds like a different
                # (female-robotic) voice and would confuse callers mid-conversation.
                # MMS only falls back to edge/gtts (same "local" tier).
                if tts_engine == "mms":
                    fallbacks = ["edge", "gtts"]
                else:
                    fallbacks = [e for e in ["edge", "gtts"] if e != tts_engine]
                self._pashto_engines: list[str] = [tts_engine] + fallbacks
            else:
                raw = config.tts.pashto_engine_priority
                self._pashto_engines = [e.strip() for e in raw.split(",") if e.strip()]
            logger.info(
                "Pashto TTS engine priority: %s (voice=%s, requested=%s)",
                self._pashto_engines, voice, tts_engine,
                extra={"session_id": session_id},
            )
        else:
            self._pashto_engines = []

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

        if self._language == "pashto":
            return await self._synthesize_pashto(text)

        # Dari: ElevenLabs primary → MMS Afghan Dari fallback → silence
        return await self._synthesize_dari(text)

    # ------------------------------------------------------------------
    # Dari — ElevenLabs primary, MMS Afghan Dari fallback
    # ------------------------------------------------------------------

    async def _synthesize_dari(self, text: str) -> bool:
        """Try ElevenLabs first, fall back to MMS Afghan Dari (facebook/mms-tts-prs)."""
        if await self._synthesize_elevenlabs(text):
            return True
        logger.info(
            "Dari: ElevenLabs failed, falling back to MMS Afghan Dari",
            extra={"session_id": self.session_id},
        )
        if await self._synthesize_mms(text):
            return True
        return await self._synthesize_silence(text)

    # ------------------------------------------------------------------
    # Pashto — walk the engine priority list
    # ------------------------------------------------------------------

    async def _synthesize_pashto(self, text: str) -> bool:
        """Try each configured Pashto TTS engine in order."""
        for engine in self._pashto_engines:
            if self._cancel_event.is_set():
                return False

            logger.info(
                "Pashto TTS trying engine: %s", engine,
                extra={"session_id": self.session_id},
            )

            if engine == "mms":
                result = await self._synthesize_mms(text)
            elif engine == "elevenlabs":
                result = await self._synthesize_elevenlabs(text)
            elif engine == "narakeet":
                result = await self._synthesize_narakeet(text)
            elif engine == "micmonster":
                result = await self._synthesize_micmonster(text)
            elif engine == "speakatoo":
                result = await self._synthesize_speakatoo(text)
            elif engine == "edge":
                result = await self._synthesize_edge_tts(text)
            elif engine == "gtts":
                result = await self._synthesize_gtts(text)
            else:
                logger.warning(
                    "Unknown Pashto TTS engine '%s', skipping", engine,
                    extra={"session_id": self.session_id},
                )
                continue

            if result:
                return True
            # engine failed → try next

        logger.error(
            "All Pashto TTS engines exhausted — falling back to silence",
            extra={"session_id": self.session_id},
        )
        return await self._synthesize_silence(text)

    # ------------------------------------------------------------------
    # MMS-TTS  (primary — VITS, local GPU)
    # ------------------------------------------------------------------

    async def _synthesize_mms(self, text: str) -> bool:
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
            return False

        audio_f32 = _resample(audio_f32, self._mms_native_rate, TTS_RATE)
        audio_f32 = _apply_fade(audio_f32, TTS_RATE)
        pcm_bytes  = _pcm_to_int16(audio_f32)
        return await self._stream_pcm(pcm_bytes)

    # ------------------------------------------------------------------
    # ElevenLabs  (API-based — high quality, multilingual)
    # ------------------------------------------------------------------

    async def _synthesize_elevenlabs(self, text: str) -> bool:
        """
        ElevenLabs REST API — requires ELEVENLABS_API_KEY.
        Voice IDs for Pashto set via:
          ELEVENLABS_VOICE_ID_PASHTO_MALE   (or any multilingual voice)
          ELEVENLABS_VOICE_ID_PASHTO_FEMALE
        """
        if self._cancel_event.is_set():
            return False

        api_key = config.tts.elevenlabs_api_key
        if not api_key:
            logger.warning(
                "ElevenLabs: ELEVENLABS_API_KEY not set, skipping",
                extra={"session_id": self.session_id},
            )
            return False

        # Default ElevenLabs multilingual voice IDs used when no custom voice is
        # configured.  These are stock library voices — they work on paid plans.
        # Override by setting ELEVENLABS_VOICE_ID_PASHTO_MALE / _FEMALE in .env.
        _DEFAULT_MALE_VOICE   = "pNInz6obpgDQGcFmaJgB"   # Adam — multilingual
        _DEFAULT_FEMALE_VOICE = "EXAVITQu4vr4xnSDxMaL"   # Bella — multilingual

        voice_id = (
            self._lang_cfg.get("elevenlabs_voice_id_female", "")
            if self._voice == "female"
            else self._lang_cfg.get("elevenlabs_voice_id_male", "")
        )
        if not voice_id:
            voice_id = _DEFAULT_FEMALE_VOICE if self._voice == "female" else _DEFAULT_MALE_VOICE
            logger.info(
                "ElevenLabs: no custom voice ID set — using default %s voice (%s). "
                "Set ELEVENLABS_VOICE_ID_PASHTO_MALE / _FEMALE in .env to override.",
                self._voice,
                voice_id,
                extra={"session_id": self.session_id},
            )

        try:
            import httpx
        except ImportError:
            logger.error(
                "ElevenLabs: httpx not installed — pip install httpx",
                extra={"session_id": self.session_id},
            )
            return False

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }

        logger.info(
            "ElevenLabs: selected_voice=%s → voice_id=%s (eleven_multilingual_v2)",
            self._voice,
            voice_id,
            extra={"session_id": self.session_id},
        )
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code == 401:
                    logger.error(
                        "ElevenLabs: invalid API key (401) — check ELEVENLABS_API_KEY",
                        extra={"session_id": self.session_id},
                    )
                    return False
                if resp.status_code == 402:
                    logger.error(
                        "ElevenLabs: payment required (402) — account is on free plan "
                        "which cannot use library voices via API. "
                        "Either upgrade your plan or clone a custom voice at elevenlabs.io "
                        "and set ELEVENLABS_VOICE_ID_PASHTO_MALE in .env",
                        extra={"session_id": self.session_id},
                    )
                    return False
                if resp.status_code != 200:
                    logger.error(
                        "ElevenLabs API error %d: %s",
                        resp.status_code,
                        resp.text[:200],
                        extra={"session_id": self.session_id},
                    )
                    return False
                audio_bytes = resp.content

            if not audio_bytes:
                return False

            loop = asyncio.get_running_loop()
            pcm_bytes = await loop.run_in_executor(
                None, lambda: _mp3_bytes_to_pcm(audio_bytes, denoise=False)
            )
            if pcm_bytes is None:
                return False

            _debug_dump_audio_pair(
                provider="elevenlabs",
                session_id=self.session_id,
                source_audio=audio_bytes,
                source_ext="mp3",
                decoded_pcm=pcm_bytes,
            )

            logger.info(
                "ElevenLabs TTS success: voice=%s voice_id=%s (%d bytes)",
                self._voice,
                voice_id,
                len(audio_bytes),
                extra={"session_id": self.session_id},
            )
            return await self._stream_pcm(pcm_bytes)

        except Exception as exc:
            logger.error(
                "ElevenLabs TTS error: %s", exc,
                extra={"session_id": self.session_id},
            )
            return False

    # ------------------------------------------------------------------
    # Narakeet  (REST API — Afghan Pashto voices)
    # ------------------------------------------------------------------

    async def _synthesize_narakeet(self, text: str) -> bool:
        """
        Narakeet REST API — requires NARAKEET_API_KEY.
        Voice set via NARAKEET_VOICE_PASHTO (default: hamid).
        Afghan Pashto voices: hamid, zeba, etc.
        Docs: https://www.narakeet.com/docs/text-to-audio-api.html
        """
        if self._cancel_event.is_set():
            return False

        api_key = config.tts.narakeet_api_key
        if not api_key:
            logger.warning(
                "Narakeet: NARAKEET_API_KEY not set, skipping",
                extra={"session_id": self.session_id},
            )
            return False

        voice = self._lang_cfg.get("narakeet_voice", "hamid")

        try:
            import httpx
        except ImportError:
            logger.error(
                "Narakeet: httpx not installed — pip install httpx",
                extra={"session_id": self.session_id},
            )
            return False

        url = "https://api.narakeet.com/text-to-speech/mp3"
        headers = {
            "x-api-key": api_key,
            "Content-Type": "text/plain",
            "Accept": "application/octet-stream",
        }
        params = {"voice": voice}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    url, headers=headers, params=params, content=text.encode("utf-8")
                )
                if resp.status_code != 200:
                    logger.error(
                        "Narakeet API error %d: %s",
                        resp.status_code,
                        resp.text[:200],
                        extra={"session_id": self.session_id},
                    )
                    return False
                audio_bytes = resp.content

            if not audio_bytes:
                return False

            loop = asyncio.get_running_loop()
            pcm_bytes = await loop.run_in_executor(
                None, lambda: _mp3_bytes_to_pcm(audio_bytes, denoise=False)
            )
            if pcm_bytes is None:
                return False

            logger.info(
                "Narakeet TTS success (%d bytes audio, voice=%s)",
                len(audio_bytes), voice,
                extra={"session_id": self.session_id},
            )
            return await self._stream_pcm(pcm_bytes)

        except Exception as exc:
            logger.error(
                "Narakeet TTS error: %s", exc,
                extra={"session_id": self.session_id},
            )
            return False

    # ------------------------------------------------------------------
    # MicMonster  (REST API)
    # ------------------------------------------------------------------

    async def _synthesize_micmonster(self, text: str) -> bool:
        """
        MicMonster REST API — requires MICMONSTER_API_KEY.
        Voice ID set via MICMONSTER_VOICE_ID_PASHTO.
        Docs: https://micmonster.com/api-documentation
        """
        if self._cancel_event.is_set():
            return False

        api_key = config.tts.micmonster_api_key
        if not api_key:
            logger.warning(
                "MicMonster: MICMONSTER_API_KEY not set, skipping",
                extra={"session_id": self.session_id},
            )
            return False

        voice_id = self._lang_cfg.get("micmonster_voice_id", "")
        if not voice_id:
            logger.warning(
                "MicMonster: MICMONSTER_VOICE_ID_PASHTO not set, skipping",
                extra={"session_id": self.session_id},
            )
            return False

        try:
            import httpx
        except ImportError:
            logger.error(
                "MicMonster: httpx not installed — pip install httpx",
                extra={"session_id": self.session_id},
            )
            return False

        url = "https://api.micmonster.com/tts"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {"voice": voice_id, "text": text, "format": "mp3"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code != 200:
                    logger.error(
                        "MicMonster API error %d: %s",
                        resp.status_code,
                        resp.text[:200],
                        extra={"session_id": self.session_id},
                    )
                    return False

                # MicMonster returns JSON with audio_url or base64
                data = resp.json()
                audio_url = data.get("audio_url") or data.get("url", "")
                if audio_url:
                    audio_resp = await client.get(audio_url, timeout=30.0)
                    audio_bytes = audio_resp.content
                else:
                    import base64
                    b64 = data.get("audio", "")
                    if not b64:
                        logger.error(
                            "MicMonster: no audio in response",
                            extra={"session_id": self.session_id},
                        )
                        return False
                    audio_bytes = base64.b64decode(b64)

            if not audio_bytes:
                return False

            loop = asyncio.get_running_loop()
            pcm_bytes = await loop.run_in_executor(
                None, lambda: _mp3_bytes_to_pcm(audio_bytes, denoise=False)
            )
            if pcm_bytes is None:
                return False

            logger.info(
                "MicMonster TTS success (%d bytes audio)",
                len(audio_bytes),
                extra={"session_id": self.session_id},
            )
            return await self._stream_pcm(pcm_bytes)

        except Exception as exc:
            logger.error(
                "MicMonster TTS error: %s", exc,
                extra={"session_id": self.session_id},
            )
            return False

    # ------------------------------------------------------------------
    # Speakatoo  (REST API)
    # ------------------------------------------------------------------

    async def _synthesize_speakatoo(self, text: str) -> bool:
        """
        Speakatoo REST API — requires SPEAKATOO_API_KEY.
        Voice ID set via SPEAKATOO_VOICE_ID_PASHTO.
        Docs: https://www.speakatoo.com/api
        """
        if self._cancel_event.is_set():
            return False

        api_key = config.tts.speakatoo_api_key
        if not api_key:
            logger.warning(
                "Speakatoo: SPEAKATOO_API_KEY not set, skipping",
                extra={"session_id": self.session_id},
            )
            return False

        voice_id = self._lang_cfg.get("speakatoo_voice_id", "")
        if not voice_id:
            logger.warning(
                "Speakatoo: SPEAKATOO_VOICE_ID_PASHTO not set, skipping",
                extra={"session_id": self.session_id},
            )
            return False

        try:
            import httpx
        except ImportError:
            logger.error(
                "Speakatoo: httpx not installed — pip install httpx",
                extra={"session_id": self.session_id},
            )
            return False

        url = "https://www.speakatoo.com/api/v1/convert"
        headers = {"Content-Type": "application/json"}
        payload = {
            "api_key": api_key,
            "voice_id": voice_id,
            "content": text,
            "output_format": "mp3",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code != 200:
                    logger.error(
                        "Speakatoo API error %d: %s",
                        resp.status_code,
                        resp.text[:200],
                        extra={"session_id": self.session_id},
                    )
                    return False

                # Response may be raw MP3 bytes or JSON with URL
                content_type = resp.headers.get("content-type", "")
                if "audio" in content_type or "octet" in content_type:
                    audio_bytes = resp.content
                else:
                    data = resp.json()
                    audio_url = data.get("url") or data.get("audio_url", "")
                    if not audio_url:
                        logger.error(
                            "Speakatoo: no audio URL in response",
                            extra={"session_id": self.session_id},
                        )
                        return False
                    audio_resp = await client.get(audio_url, timeout=30.0)
                    audio_bytes = audio_resp.content

            if not audio_bytes:
                return False

            loop = asyncio.get_running_loop()
            pcm_bytes = await loop.run_in_executor(
                None, lambda: _mp3_bytes_to_pcm(audio_bytes, denoise=False)
            )
            if pcm_bytes is None:
                return False

            logger.info(
                "Speakatoo TTS success (%d bytes audio)",
                len(audio_bytes),
                extra={"session_id": self.session_id},
            )
            return await self._stream_pcm(pcm_bytes)

        except Exception as exc:
            logger.error(
                "Speakatoo TTS error: %s", exc,
                extra={"session_id": self.session_id},
            )
            return False

    # ------------------------------------------------------------------
    # edge-tts  (fallback 1 for Dari, configurable for Pashto)
    # ------------------------------------------------------------------

    async def _synthesize_edge_tts(self, text: str) -> bool:
        """Microsoft edge-tts neural voice (free, no API key required)."""
        if self._cancel_event.is_set():
            return False

        try:
            import edge_tts
            import av
        except ImportError:
            return False

        try:
            communicate = edge_tts.Communicate(text, voice=self._edge_tts_voice)
            audio_bytes = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes += chunk["data"]
                if self._cancel_event.is_set():
                    return False

            if not audio_bytes:
                return False

            loop = asyncio.get_running_loop()
            pcm_bytes = await loop.run_in_executor(
                None, lambda: _mp3_bytes_to_pcm(audio_bytes, denoise=False)
            )
            if pcm_bytes is None:
                return False

            logger.info(
                "edge-tts success (voice=%s)", self._edge_tts_voice,
                extra={"session_id": self.session_id},
            )
            return await self._stream_pcm(pcm_bytes)

        except Exception as exc:
            logger.error("edge-tts error: %s", exc,
                         extra={"session_id": self.session_id})
            return False

    # ------------------------------------------------------------------
    # gTTS  (fallback 2)
    # ------------------------------------------------------------------

    async def _synthesize_gtts(self, text: str) -> bool:
        if self._cancel_event.is_set():
            return False
        try:
            from gtts import gTTS
        except ImportError:
            return False

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
                None, lambda: _mp3_bytes_to_pcm(audio_bytes, denoise=False)
            )
            if pcm_bytes is None:
                return False

            return await self._stream_pcm(pcm_bytes)

        except Exception as exc:
            logger.error("gTTS error: %s", exc,
                         extra={"session_id": self.session_id})
            return False

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
            chunk = pcm_bytes[i: i + bpc]
            await self._send_audio(chunk)
            self.last_pcm_bytes_sent += len(chunk)
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
