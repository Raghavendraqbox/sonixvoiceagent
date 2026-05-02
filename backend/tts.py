"""
tts.py — Multi-language TTS handler for Telugu and Kannada.

Priority chain per language:
  Telugu (configurable via TELUGU_TTS_ENGINE_PRIORITY):
    Engines available:
      sarvam     — Sarvam AI REST API    (requires SARVAM_API_KEY, best Telugu quality)
      google_tts — Google Cloud TTS      (requires GOOGLE_APPLICATION_CREDENTIALS)
      gnani      — Gnani.ai REST API     (requires GNANI_API_KEY + GNANI_CLIENT_ID)
      ttsmaker   — TTSMaker REST API     (requires TTSMAKER_TOKEN + TTSMAKER_VOICE_ID_TELUGU)
      elevenlabs — ElevenLabs REST API   (requires ELEVENLABS_API_KEY)
      edge       — Microsoft edge-tts    (free, te-IN-ShrutiNeural / MohanNeural)
      gtts       — Google gTTS           (fallback, uses Telugu te)

    Default priority: sarvam,google_tts,gnani,ttsmaker,edge,gtts
    Override via env: TELUGU_TTS_ENGINE_PRIORITY=sarvam,edge,gtts

  Kannada (configurable via KANNADA_TTS_ENGINE_PRIORITY):
    Engines available:
      sarvam       — Sarvam AI REST API    (requires SARVAM_API_KEY, best Indian language quality)
      google_tts   — Google Cloud TTS      (requires GOOGLE_APPLICATION_CREDENTIALS)
      gnani        — Gnani.ai REST API     (requires GNANI_API_KEY + GNANI_CLIENT_ID)
      ttsmaker     — TTSMaker REST API     (requires TTSMAKER_TOKEN + TTSMAKER_VOICE_ID_KANNADA)
      elevenlabs   — ElevenLabs REST API   (requires ELEVENLABS_API_KEY)
      azure_tts    — Microsoft Azure TTS   (requires AZURE_TTS_KEY, kn-IN-SapnaNeural / GaganNeural)
      amazon_polly — Amazon Polly          (requires AWS credentials, no native kn voice)
      mms          — facebook/mms-tts-kan  (local GPU)
      narakeet     — Narakeet REST API     (requires NARAKEET_API_KEY)
      micmonster   — MicMonster REST API   (requires MICMONSTER_API_KEY)
      speakatoo    — Speakatoo REST API    (requires SPEAKATOO_API_KEY)
      edge         — Microsoft edge-tts    (free, kn-IN-SapnaNeural / GaganNeural)
      gtts         — Google gTTS           (fallback, uses Kannada kn)

    Default priority: sarvam,google_tts,gnani,ttsmaker,elevenlabs,edge,gtts
    Override via env: KANNADA_TTS_ENGINE_PRIORITY=sarvam,edge,gtts

Audio output contract:
  PCM 16-bit signed LE, 24 000 Hz, mono
  Must match PLAYBACK_SAMPLE_RATE = 24000 in frontend/index.html

Cancel semantics:
  Setting cancel_event aborts synthesis between 60 ms chunks.
"""

import asyncio
import contextvars
import datetime
import io
import logging
import os
import threading
import wave
from pathlib import Path
from typing import Callable, Awaitable, Optional

import numpy as np

import httpx

from config import (
    SARVAM_EMOTION_PRESETS,
    SARVAM_EMOTION_TEMPERATURES,
    SARVAM_FEMALE_SPEAKERS,
    config,
    get_language_config,
)

# Sarvam Bulbul v3 strict speaker whitelist. The lists below are taken from
# the API's own 400-error response and the public Bulbul v3 docs — anything
# outside these sets is rejected by /text-to-speech and is remapped to the
# default v3 speaker for the requested gender at synthesis time.
BULBUL_V3_FEMALE_SPEAKERS = {
    "priya", "ritu", "neha", "pooja", "simran", "kavya",
    "ishita", "shreya", "roopa", "tanya", "shruti", "suhani",
    "kavitha", "rupali",
}
BULBUL_V3_MALE_SPEAKERS = {
    "shubh", "aditya", "rahul", "rohan", "amit", "dev", "ratan",
    "varun", "manan", "sumit", "kabir", "aayan", "ashutosh",
    "advait", "anand", "tarun", "sunny", "mani", "gokul", "vijay",
    "mohit", "rehan", "soham",
}
# "priya" / "aditya" have the lowest CER in Sarvam's published quality
# rankings, making them the safest defaults when the requested speaker is
# not v3-compatible.
_BULBUL_V3_DEFAULT_FEMALE = "priya"
_BULBUL_V3_DEFAULT_MALE = "aditya"

logger = logging.getLogger(__name__)

# Persistent HTTP client reused across all cloud TTS calls — eliminates
# ~150-200ms TCP+TLS reconnection overhead per sentence synthesis.
_cloud_http: Optional[httpx.AsyncClient] = None
_google_tts_client = None
_google_tts_lock = threading.Lock()


def _get_cloud_client() -> httpx.AsyncClient:
    """Return the shared persistent HTTP client, creating it on first call."""
    global _cloud_http
    if _cloud_http is None or _cloud_http.is_closed:
        _cloud_http = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=5, keepalive_expiry=120),
        )
    return _cloud_http


def _get_google_tts_client():
    """Return the Google Cloud TTS SDK client, using ADC from the environment."""
    global _google_tts_client
    if _google_tts_client is not None:
        return _google_tts_client

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not credentials_path:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is not set")
    if not Path(credentials_path).expanduser().is_file():
        raise RuntimeError(f"GOOGLE_APPLICATION_CREDENTIALS file not found: {credentials_path}")

    with _google_tts_lock:
        if _google_tts_client is None:
            from google.cloud import texttospeech

            _google_tts_client = texttospeech.TextToSpeechClient()
    return _google_tts_client


TTS_RATE = config.tts.sample_rate      # 24 000 Hz — browser playback rate

AudioSendCallback = Callable[[bytes], Awaitable[None]]


def _normalize_sarvam_emotion(emotion: str) -> str:
    normalized = (emotion or "neutral").strip().lower().replace("-", "_")
    return normalized if normalized in SARVAM_EMOTION_PRESETS else "neutral"


def _parse_sarvam_temperature(raw: str) -> Optional[float]:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return min(2.0, max(0.01, value))


def _clamp_sarvam_pace(value: float) -> float:
    """Clamp a pace value to Sarvam's supported [0.5, 2.0] range."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return config.tts.sarvam_pace
    return max(0.5, min(2.0, v))

# Task-local capture buffer: when set, _stream_pcm appends bytes here instead
# of sending to the client. Used by synthesize_to_pcm for pipeline pre-synthesis.
# asyncio task contexts are isolated, so concurrent synthesis + streaming never
# interfere even when both call _stream_pcm on the same VoiceTTSHandler.
_pcm_capture_var: contextvars.ContextVar[Optional[list]] = contextvars.ContextVar(
    "_tts_pcm_capture", default=None
)


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


def schedule_tts_warmup(language: str = "telugu") -> None:
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


async def warmup_tts_connection() -> None:
    """
    Pre-establish TLS connections to configured cloud TTS endpoints.

    Eliminates the ~150-200ms TCP+TLS handshake cost from the very first
    synthesis request per call. Called once at startup after LLM warmup.
    """
    endpoints: list[str] = []
    if config.tts.azure_tts_key:
        endpoints.append(
            f"https://{config.tts.azure_tts_region}.tts.speech.microsoft.com"
        )
    if config.tts.sarvam_api_key:
        endpoints.append("https://api.sarvam.ai")
    if not endpoints:
        return
    client = _get_cloud_client()
    for url in endpoints:
        try:
            await client.get(url, timeout=5.0)
        except Exception:
            pass  # Any response (even 4xx) means TLS session is established
    logger.info("TTS connections pre-warmed (%d endpoint(s))", len(endpoints))


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
    Synthesizes Telugu or Kannada text to PCM audio and streams it chunk-by-chunk.

    Telugu (order set by TELUGU_TTS_ENGINE_PRIORITY):
      sarvam, google_tts, gnani, ttsmaker, elevenlabs, azure_tts, amazon_polly, edge, gtts

    Kannada (order set by KANNADA_TTS_ENGINE_PRIORITY):
      sarvam, google_tts, gnani, ttsmaker, elevenlabs, azure_tts, amazon_polly,
      mms, narakeet, micmonster, speakatoo, edge, gtts
    """

    def __init__(
        self,
        session_id: str,
        send_audio_cb: AudioSendCallback,
        cancel_event: asyncio.Event,
        language: str = "telugu",
        voice: str = "male",
        tts_engine: str = "auto",
        sarvam_speaker: str = "",
        sarvam_emotion: str = "",
        sarvam_pace: float = 0.0,
    ) -> None:
        self.session_id    = session_id
        self._send_audio   = send_audio_cb
        self._cancel_event = cancel_event
        self._language     = language.lower()
        self._voice        = voice.lower()
        self._sarvam_speaker_override = sarvam_speaker.strip().lower()
        requested_emotion = sarvam_emotion or config.tts.sarvam_emotion
        self._sarvam_emotion = _normalize_sarvam_emotion(requested_emotion)
        if requested_emotion and self._sarvam_emotion != requested_emotion.strip().lower().replace("-", "_"):
            logger.warning(
                "Ignoring unsupported Sarvam emotion '%s'",
                requested_emotion,
                extra={"session_id": session_id},
            )
        emotion_preset = SARVAM_EMOTION_PRESETS[self._sarvam_emotion]
        self._sarvam_temperature = float(emotion_preset["temperature"])
        self._sarvam_emotion_pace_offset = float(emotion_preset["pace_offset"])
        temperature_override = _parse_sarvam_temperature(config.tts.sarvam_temperature)
        if temperature_override is not None:
            self._sarvam_temperature = temperature_override
        # Per-session base pace from the frontend slider; falls back to the
        # SARVAM_PACE env default when the client did not pass one.
        base_pace = sarvam_pace if sarvam_pace and sarvam_pace > 0 else config.tts.sarvam_pace
        self._sarvam_base_pace = _clamp_sarvam_pace(base_pace)
        self.last_pcm_bytes_sent: int = 0  # tracks bytes sent in last synthesis

        if (
            self._sarvam_speaker_override
            and self._sarvam_speaker_override not in SARVAM_FEMALE_SPEAKERS
        ):
            logger.warning(
                "Ignoring unsupported Sarvam female speaker '%s'",
                self._sarvam_speaker_override,
                extra={"session_id": session_id},
            )
            self._sarvam_speaker_override = ""

        lang_cfg = get_language_config(language)
        self._mms_model_id: str     = lang_cfg["mms_tts_model"]
        self._mms_native_rate: int  = lang_cfg["mms_tts_sample_rate"]
        self._gtts_language: str    = lang_cfg["gtts_language"]
        self._language_display: str = lang_cfg["display_name"]
        self._lang_cfg              = lang_cfg

        # Telugu must remain strict Telugu-only (MMS model). Keep voice metadata for
        # Kannada/cloud providers, but never switch Telugu to non-Telugu engines.
        if voice == "female":
            self._edge_tts_voice: str       = lang_cfg["edge_tts_voice"]
            self._use_edge_primary: bool    = False
        else:
            self._edge_tts_voice: str       = lang_cfg.get("edge_tts_voice_male", lang_cfg["edge_tts_voice"])
            self._use_edge_primary: bool    = False

        # Build Telugu engine priority list
        # "auto" → use TELUGU_TTS_ENGINE_PRIORITY from .env
        if self._language == "telugu":
            TELUGU_VALID = {"sarvam", "google_tts", "gnani", "ttsmaker", "elevenlabs", "azure_tts", "amazon_polly", "edge", "gtts"}
            if tts_engine and tts_engine != "auto" and tts_engine in TELUGU_VALID:
                fallbacks = [e for e in ["edge", "gtts"] if e != tts_engine]
                self._telugu_engines: list[str] = [tts_engine] + fallbacks
            else:
                raw = config.tts.telugu_engine_priority
                self._telugu_engines = [e.strip() for e in raw.split(",") if e.strip()]
            logger.info(
                "Telugu TTS engine priority: %s (voice=%s, requested=%s)",
                self._telugu_engines, voice, tts_engine,
                extra={"session_id": session_id},
            )
        else:
            self._telugu_engines = []

        # Build Kannada engine priority list
        # If a specific engine is requested from the UI, put it first with fallbacks.
        # "auto" → use KANNADA_TTS_ENGINE_PRIORITY from .env
        if self._language == "kannada":
            VALID_ENGINES = {
                "sarvam", "google_tts", "gnani", "ttsmaker",
                "elevenlabs", "azure_tts", "amazon_polly",
                "mms", "narakeet", "micmonster", "speakatoo", "edge", "gtts",
            }
            if tts_engine and tts_engine != "auto" and tts_engine in VALID_ENGINES:
                # MMS falls back to edge/gtts; all others fall back to edge/gtts too.
                fallbacks = [e for e in ["edge", "gtts"] if e != tts_engine]
                self._kannada_engines: list[str] = [tts_engine] + fallbacks
            else:
                raw = config.tts.kannada_engine_priority
                self._kannada_engines = [e.strip() for e in raw.split(",") if e.strip()]
            logger.info(
                "Kannada TTS engine priority: %s (voice=%s, requested=%s)",
                self._kannada_engines, voice, tts_engine,
                extra={"session_id": session_id},
            )
        else:
            self._kannada_engines = []

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

        if self._language == "kannada":
            return await self._synthesize_kannada(text)

        # Telugu: ElevenLabs primary → MMS Telugu fallback → silence
        return await self._synthesize_telugu(text)

    async def synthesize_to_pcm(self, text: str) -> Optional[bytes]:
        """Synthesize text → raw PCM bytes without streaming to the client.

        Sets _pcm_capture_var in the current task context so _stream_pcm
        captures bytes instead of sending. Safe to call concurrently with
        _stream_pcm on the same handler because task contexts are isolated.
        """
        if not text.strip():
            return None
        capture: list[bytes] = []
        token = _pcm_capture_var.set(capture)
        try:
            await self.synthesize_and_stream(text)
        finally:
            _pcm_capture_var.reset(token)
        return b"".join(capture) if capture else None

    # ------------------------------------------------------------------
    # Telugu — walk the configurable engine priority list
    # ------------------------------------------------------------------

    async def _synthesize_telugu(self, text: str) -> bool:
        """Try each configured Telugu TTS engine in order."""
        for engine in self._telugu_engines:
            if self._cancel_event.is_set():
                return False

            logger.info(
                "Telugu TTS trying engine: %s", engine,
                extra={"session_id": self.session_id},
            )

            if engine == "sarvam":
                result = await self._synthesize_sarvam(text)
            elif engine == "google_tts":
                result = await self._synthesize_google_tts(text)
            elif engine == "gnani":
                result = await self._synthesize_gnani(text)
            elif engine == "ttsmaker":
                result = await self._synthesize_ttsmaker(text)
            elif engine == "elevenlabs":
                result = await self._synthesize_elevenlabs(text)
            elif engine == "azure_tts":
                result = await self._synthesize_azure_tts(text)
            elif engine == "amazon_polly":
                result = await self._synthesize_amazon_polly(text)
            elif engine == "edge":
                result = await self._synthesize_edge_tts(text)
            elif engine == "gtts":
                result = await self._synthesize_gtts(text)
            else:
                logger.warning(
                    "Unknown Telugu TTS engine '%s', skipping", engine,
                    extra={"session_id": self.session_id},
                )
                continue

            if result:
                return True
            # engine failed → try next

        logger.error(
            "All Telugu TTS engines exhausted — falling back to silence",
            extra={"session_id": self.session_id},
        )
        return await self._synthesize_silence(text)

    # ------------------------------------------------------------------
    # Kannada — walk the engine priority list
    # ------------------------------------------------------------------

    async def _synthesize_kannada(self, text: str) -> bool:
        """Try each configured Kannada TTS engine in order."""
        for engine in self._kannada_engines:
            if self._cancel_event.is_set():
                return False

            logger.info(
                "Kannada TTS trying engine: %s", engine,
                extra={"session_id": self.session_id},
            )

            if engine == "sarvam":
                result = await self._synthesize_sarvam(text)
            elif engine == "google_tts":
                result = await self._synthesize_google_tts(text)
            elif engine == "gnani":
                result = await self._synthesize_gnani(text)
            elif engine == "ttsmaker":
                result = await self._synthesize_ttsmaker(text)
            elif engine == "elevenlabs":
                result = await self._synthesize_elevenlabs(text)
            elif engine == "azure_tts":
                result = await self._synthesize_azure_tts(text)
            elif engine == "amazon_polly":
                result = await self._synthesize_amazon_polly(text)
            elif engine == "mms":
                result = await self._synthesize_mms(text)
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
                    "Unknown Kannada TTS engine '%s', skipping", engine,
                    extra={"session_id": self.session_id},
                )
                continue

            if result:
                return True
            # engine failed → try next

        logger.error(
            "All Kannada TTS engines exhausted — falling back to silence",
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
    # Sarvam AI  (best quality for Indian languages including Telugu)
    # Docs: https://docs.sarvam.ai/api-reference-docs/text-to-speech
    # ------------------------------------------------------------------

    async def _synthesize_sarvam(self, text: str) -> bool:
        """
        Sarvam AI TTS — best quality for Telugu.
        Requires SARVAM_API_KEY.
        Speaker overrides: SARVAM_SPEAKER_TELUGU / SARVAM_SPEAKER_TELUGU_MALE
        Model override: SARVAM_MODEL (default: bulbul:v3)
        """
        if self._cancel_event.is_set():
            return False

        api_key = config.tts.sarvam_api_key
        if not api_key:
            logger.warning(
                "Sarvam AI: SARVAM_API_KEY not set, skipping",
                extra={"session_id": self.session_id},
            )
            return False

        if self._voice == "female":
            speaker = self._sarvam_speaker_override or self._lang_cfg.get(
                "sarvam_speaker", _BULBUL_V3_DEFAULT_FEMALE
            )
        else:
            speaker = self._lang_cfg.get("sarvam_speaker_male", _BULBUL_V3_DEFAULT_MALE)
        language_code = self._lang_cfg.get("sarvam_language_code", "te-IN")
        model         = self._lang_cfg.get("sarvam_model", "bulbul:v3")
        is_bulbul_v3  = str(model).lower().startswith("bulbul:v3")
        if is_bulbul_v3:
            valid_speakers = (
                BULBUL_V3_FEMALE_SPEAKERS
                if self._voice == "female"
                else BULBUL_V3_MALE_SPEAKERS
            )
            default_speaker = (
                _BULBUL_V3_DEFAULT_FEMALE
                if self._voice == "female"
                else _BULBUL_V3_DEFAULT_MALE
            )
            if speaker not in valid_speakers:
                logger.info(
                    "Sarvam AI: speaker '%s' is not in the bulbul:v3 whitelist — "
                    "using '%s' instead",
                    speaker,
                    default_speaker,
                    extra={"session_id": self.session_id},
                )
                speaker = default_speaker

        # Per-emotion pace offset is added to the user-selected slider value
        # so calm vs excited differ in both temperature and speed.
        effective_pace = _clamp_sarvam_pace(
            self._sarvam_base_pace + self._sarvam_emotion_pace_offset
        )

        url = "https://api.sarvam.ai/text-to-speech"
        headers = {
            "API-Subscription-Key": api_key,
            "Content-Type": "application/json",
        }
        # Sarvam accepts up to 500 chars per request; split if needed
        payload = {
            "inputs":               [text],
            "target_language_code": language_code,
            "speaker":              speaker,
            "pace":                 effective_pace,
            "speech_sample_rate":   22050,
            "model":                model,
        }
        if is_bulbul_v3:
            payload["temperature"] = self._sarvam_temperature
        else:
            payload["enable_preprocessing"] = True
            payload["pitch"] = 0
            payload["loudness"] = 1.5

        try:
            client = _get_cloud_client()
            resp = await client.post(url, headers=headers, json=payload)
            if (
                resp.status_code == 400
                and is_bulbul_v3
                and "not compatible with model bulbul:v3" in resp.text
            ):
                fallback_speaker = (
                    _BULBUL_V3_DEFAULT_FEMALE
                    if self._voice == "female"
                    else _BULBUL_V3_DEFAULT_MALE
                )
                if payload.get("speaker") != fallback_speaker:
                    logger.warning(
                        "Sarvam AI: speaker '%s' incompatible with bulbul:v3, retrying with '%s'",
                        payload.get("speaker"),
                        fallback_speaker,
                        extra={"session_id": self.session_id},
                    )
                    payload["speaker"] = fallback_speaker
                    resp = await client.post(url, headers=headers, json=payload)
            if (
                resp.status_code in (400, 422)
                and is_bulbul_v3
                and "temperature" in resp.text.lower()
                and "temperature" in payload
            ):
                logger.warning(
                    "Sarvam AI: temperature rejected for bulbul:v3, retrying without emotion preset",
                    extra={"session_id": self.session_id},
                )
                payload.pop("temperature", None)
                resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code == 401:
                logger.error("Sarvam AI: invalid API key (401)",
                             extra={"session_id": self.session_id})
                return False
            if resp.status_code != 200:
                logger.error(
                    "Sarvam AI API error %d: %s",
                    resp.status_code, resp.text[:200],
                    extra={"session_id": self.session_id},
                )
                return False
            data = resp.json()

            audios = data.get("audios") or []
            if not audios:
                logger.error("Sarvam AI: empty 'audios' in response",
                             extra={"session_id": self.session_id})
                return False

            import base64
            wav_bytes = base64.b64decode(audios[0])
            if not wav_bytes:
                return False

            # WAV is decoded via PyAV (same as MP3 — av.open auto-detects format)
            loop = asyncio.get_running_loop()
            pcm_bytes = await loop.run_in_executor(
                None, lambda: _mp3_bytes_to_pcm(wav_bytes, denoise=False)
            )
            if pcm_bytes is None:
                return False

            effective_speaker = payload.get("speaker", speaker)
            # `temperature` may have been popped by the retry path — log
            # exactly what the API used so it's obvious when an emotion
            # preset silently failed and the synthesis fell back to Sarvam's
            # internal default.
            applied_temperature = payload.get("temperature")
            temperature_repr = (
                f"{applied_temperature:.2f}"
                if applied_temperature is not None
                else "default(stripped)"
            )
            _debug_dump_audio_pair(
                provider="sarvam",
                session_id=self.session_id,
                source_audio=wav_bytes,
                source_ext="wav",
                decoded_pcm=pcm_bytes,
            )
            logger.info(
                "Sarvam AI TTS success: speaker=%s language=%s model=%s emotion=%s "
                "temperature=%s pace=%.2f (base=%.2f + offset=%+.2f) (%d bytes)",
                effective_speaker,
                language_code,
                model,
                self._sarvam_emotion,
                temperature_repr,
                effective_pace,
                self._sarvam_base_pace,
                self._sarvam_emotion_pace_offset,
                len(wav_bytes),
                extra={"session_id": self.session_id},
            )
            return await self._stream_pcm(pcm_bytes)

        except Exception as exc:
            logger.error("Sarvam AI TTS error: %s", exc,
                         extra={"session_id": self.session_id})
            return False

    # ------------------------------------------------------------------
    # Google Cloud TTS  (https://cloud.google.com/text-to-speech)
    # ------------------------------------------------------------------

    async def _synthesize_google_tts(self, text: str) -> bool:
        """
        Google Cloud TTS — uses the official SDK with GOOGLE_APPLICATION_CREDENTIALS.
        Voice overrides: GOOGLE_TTS_VOICE_TELUGU / GOOGLE_TTS_VOICE_TELUGU_MALE
        Telugu voices: te-IN-Standard-A (female), te-IN-Standard-B (male)
                       te-IN-Wavenet-A  (female), te-IN-Wavenet-B  (male)
        """
        if self._cancel_event.is_set():
            return False

        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
        if not credentials_path:
            logger.warning(
                "Google TTS: GOOGLE_APPLICATION_CREDENTIALS not set, skipping",
                extra={"session_id": self.session_id},
            )
            return False
        if not Path(credentials_path).expanduser().is_file():
            logger.error(
                "Google TTS: GOOGLE_APPLICATION_CREDENTIALS file not found: %s",
                credentials_path,
                extra={"session_id": self.session_id},
            )
            return False

        voice_name = (
            self._lang_cfg.get("google_tts_voice", "te-IN-Standard-A")
            if self._voice == "female"
            else self._lang_cfg.get("google_tts_voice_male", "te-IN-Standard-B")
        )
        language_code = self._lang_cfg.get("google_tts_language", "te-IN")

        try:
            from google.cloud import texttospeech

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: _get_google_tts_client().synthesize_speech(
                    input=texttospeech.SynthesisInput(text=text),
                    voice=texttospeech.VoiceSelectionParams(
                        language_code=language_code,
                        name=voice_name,
                    ),
                    audio_config=texttospeech.AudioConfig(
                        audio_encoding=texttospeech.AudioEncoding.MP3,
                        sample_rate_hertz=TTS_RATE,
                    ),
                ),
            )

            mp3_bytes = response.audio_content
            if not mp3_bytes:
                logger.error(
                    "Google TTS: empty audio_content in response",
                    extra={"session_id": self.session_id},
                )
                return False

            pcm_bytes = await loop.run_in_executor(
                None, lambda: _mp3_bytes_to_pcm(mp3_bytes, denoise=False)
            )
            if pcm_bytes is None:
                return False

            _debug_dump_audio_pair(
                provider="google_tts",
                session_id=self.session_id,
                source_audio=mp3_bytes,
                source_ext="mp3",
                decoded_pcm=pcm_bytes,
            )
            logger.info(
                "Google TTS success: voice=%s (%d bytes)",
                voice_name, len(mp3_bytes),
                extra={"session_id": self.session_id},
            )
            return await self._stream_pcm(pcm_bytes)

        except Exception as exc:
            logger.error("Google TTS error: %s", exc,
                         extra={"session_id": self.session_id})
            return False

    # ------------------------------------------------------------------
    # Gnani.ai  (https://gnani.ai — Indian language specialist)
    # ------------------------------------------------------------------

    async def _synthesize_gnani(self, text: str) -> bool:
        """
        Gnani.ai TTS — requires GNANI_API_KEY (and optionally GNANI_CLIENT_ID).
        Voice override: GNANI_VOICE_TELUGU (default: 'female')
        Contact Gnani.ai for your API key and voice IDs.
        """
        if self._cancel_event.is_set():
            return False

        api_key   = config.tts.gnani_api_key
        client_id = config.tts.gnani_client_id
        if not api_key:
            logger.warning(
                "Gnani.ai: GNANI_API_KEY not set, skipping",
                extra={"session_id": self.session_id},
            )
            return False

        language_code = self._lang_cfg.get("gnani_language_code", "te")
        voice         = self._lang_cfg.get("gnani_voice", "female")

        # Gnani.ai REST TTS endpoint
        url = "https://dev.gnani.ai/api/v1/synthesize"
        headers: dict = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }
        if client_id:
            headers["X-Client-Id"] = client_id

        payload = {
            "text":     text,
            "language": language_code,
            "voice":    voice,
            "format":   "mp3",
        }

        try:
            client = _get_cloud_client()
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code in (401, 403):
                logger.error(
                    "Gnani.ai: auth error (%d) — check GNANI_API_KEY / GNANI_CLIENT_ID",
                    resp.status_code,
                    extra={"session_id": self.session_id},
                )
                return False
            if resp.status_code != 200:
                logger.error(
                    "Gnani.ai API error %d: %s",
                    resp.status_code, resp.text[:200],
                    extra={"session_id": self.session_id},
                )
                return False

            # Gnani returns raw MP3 bytes (content-type: audio/mpeg)
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
                provider="gnani",
                session_id=self.session_id,
                source_audio=audio_bytes,
                source_ext="mp3",
                decoded_pcm=pcm_bytes,
            )
            logger.info(
                "Gnani.ai TTS success: language=%s voice=%s (%d bytes)",
                language_code, voice, len(audio_bytes),
                extra={"session_id": self.session_id},
            )
            return await self._stream_pcm(pcm_bytes)

        except Exception as exc:
            logger.error("Gnani.ai TTS error: %s", exc,
                         extra={"session_id": self.session_id})
            return False

    # ------------------------------------------------------------------
    # TTSMaker  (https://ttsmaker.com — free tier available)
    # Docs: https://ttsmaker.com/api-doc
    # ------------------------------------------------------------------

    async def _synthesize_ttsmaker(self, text: str) -> bool:
        """
        TTSMaker TTS — requires TTSMAKER_TOKEN and TTSMAKER_VOICE_ID_TELUGU.
        Get your token and browse Telugu voice IDs at: https://ttsmaker.com
        API docs: https://api.ttsmaker.com/v1/get-voice-list  (voice_id lookup)
        """
        if self._cancel_event.is_set():
            return False

        token    = config.tts.ttsmaker_token
        voice_id = self._lang_cfg.get("ttsmaker_voice_id", 0)
        if not token:
            logger.warning(
                "TTSMaker: TTSMAKER_TOKEN not set, skipping",
                extra={"session_id": self.session_id},
            )
            return False
        if not voice_id:
            logger.warning(
                "TTSMaker: TTSMAKER_VOICE_ID_TELUGU not set (must be a non-zero int), skipping",
                extra={"session_id": self.session_id},
            )
            return False

        order_url = "https://api.ttsmaker.com/v1/create-tts-order"
        payload = {
            "token":                    token,
            "text":                     text,
            "voice_id":                 voice_id,
            "audio_format":             "mp3",
            "audio_speed":              1.0,
            "audio_volume":             0,
            "text_paragraph_pause_time": 0,
        }

        try:
            client = _get_cloud_client()
            # Step 1: create the TTS order (longer timeout for synthesis)
            resp = await client.post(order_url, json=payload, timeout=60.0)
            if resp.status_code != 200:
                logger.error(
                    "TTSMaker order error %d: %s",
                    resp.status_code, resp.text[:200],
                    extra={"session_id": self.session_id},
                )
                return False
            data = resp.json()

            if data.get("status") != 200:
                logger.error(
                    "TTSMaker: order failed — %s", data,
                    extra={"session_id": self.session_id},
                )
                return False

            audio_url = data.get("audio_file_url", "")
            if not audio_url:
                logger.error("TTSMaker: no audio_file_url in response",
                             extra={"session_id": self.session_id})
                return False

            # Step 2: download the generated MP3
            dl_resp = await client.get(audio_url, timeout=30.0)
            if dl_resp.status_code != 200:
                logger.error(
                    "TTSMaker: download error %d for %s",
                    dl_resp.status_code, audio_url,
                    extra={"session_id": self.session_id},
                )
                return False
            mp3_bytes = dl_resp.content

            if not mp3_bytes:
                return False

            loop = asyncio.get_running_loop()
            pcm_bytes = await loop.run_in_executor(
                None, lambda: _mp3_bytes_to_pcm(mp3_bytes, denoise=False)
            )
            if pcm_bytes is None:
                return False

            _debug_dump_audio_pair(
                provider="ttsmaker",
                session_id=self.session_id,
                source_audio=mp3_bytes,
                source_ext="mp3",
                decoded_pcm=pcm_bytes,
            )
            logger.info(
                "TTSMaker TTS success: voice_id=%s (%d bytes)",
                voice_id, len(mp3_bytes),
                extra={"session_id": self.session_id},
            )
            return await self._stream_pcm(pcm_bytes)

        except Exception as exc:
            logger.error("TTSMaker TTS error: %s", exc,
                         extra={"session_id": self.session_id})
            return False

    # ------------------------------------------------------------------
    # ElevenLabs  (API-based — high quality, multilingual)
    # ------------------------------------------------------------------

    async def _synthesize_elevenlabs(self, text: str) -> bool:
        """
        ElevenLabs REST API — requires ELEVENLABS_API_KEY.
        Voice IDs set via:
          ELEVENLABS_VOICE_ID_TELUGU_MALE / ELEVENLABS_VOICE_ID_TELUGU_FEMALE
          ELEVENLABS_VOICE_ID_KANNADA_MALE / ELEVENLABS_VOICE_ID_KANNADA_FEMALE
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
        # Override by setting ELEVENLABS_VOICE_ID_TELUGU_MALE / _FEMALE or
        # ELEVENLABS_VOICE_ID_KANNADA_MALE / _FEMALE in .env.
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
                "Set ELEVENLABS_VOICE_ID_TELUGU_MALE/_FEMALE or ELEVENLABS_VOICE_ID_KANNADA_MALE/_FEMALE in .env to override.",
                self._voice,
                voice_id,
                extra={"session_id": self.session_id},
            )

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
            client = _get_cloud_client()
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
                    "and set ELEVENLABS_VOICE_ID_TELUGU_MALE or ELEVENLABS_VOICE_ID_KANNADA_MALE in .env",
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
    # Narakeet  (REST API — Kannada voices)
    # ------------------------------------------------------------------

    async def _synthesize_narakeet(self, text: str) -> bool:
        """
        Narakeet REST API — requires NARAKEET_API_KEY.
        Voice set via NARAKEET_VOICE_KANNADA.
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

        url = "https://api.narakeet.com/text-to-speech/mp3"
        headers = {
            "x-api-key": api_key,
            "Content-Type": "text/plain",
            "Accept": "application/octet-stream",
        }
        params = {"voice": voice}

        try:
            client = _get_cloud_client()
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

        url = "https://api.micmonster.com/tts"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {"voice": voice_id, "text": text, "format": "mp3"}

        try:
            client = _get_cloud_client()
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

        url = "https://www.speakatoo.com/api/v1/convert"
        headers = {"Content-Type": "application/json"}
        payload = {
            "api_key": api_key,
            "voice_id": voice_id,
            "content": text,
            "output_format": "mp3",
        }

        try:
            client = _get_cloud_client()
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
    # Microsoft Azure Cognitive Services TTS
    # Docs: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/rest-text-to-speech
    # Telugu neural voices: te-IN-ShrutiNeural (F), te-IN-MohanNeural (M)
    # Requires: AZURE_TTS_KEY, AZURE_TTS_REGION (default: eastus)
    # ------------------------------------------------------------------

    async def _synthesize_azure_tts(self, text: str) -> bool:
        if self._cancel_event.is_set():
            return False

        api_key = config.tts.azure_tts_key
        region  = config.tts.azure_tts_region
        if not api_key:
            logger.warning(
                "Azure TTS: AZURE_TTS_KEY not set, skipping",
                extra={"session_id": self.session_id},
            )
            return False

        voice_name    = (
            self._lang_cfg.get("azure_tts_voice", "te-IN-ShrutiNeural")
            if self._voice == "female"
            else self._lang_cfg.get("azure_tts_voice_male", "te-IN-MohanNeural")
        )
        language_code = self._lang_cfg.get("azure_tts_language", "te-IN")

        import xml.sax.saxutils as saxutils
        rate = config.tts.tts_rate
        ssml = (
            f"<speak version='1.0' xml:lang='{language_code}'>"
            f"<voice xml:lang='{language_code}' name='{voice_name}'>"
            f"<prosody rate='{rate}'>"
            f"{saxutils.escape(text)}"
            f"</prosody></voice></speak>"
        )

        url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
        headers = {
            "Ocp-Apim-Subscription-Key": api_key,
            "Content-Type":              "application/ssml+xml",
            "X-Microsoft-OutputFormat":  "audio-24khz-96kbitrate-mono-mp3",
        }

        try:
            client = _get_cloud_client()
            resp = await client.post(url, headers=headers, content=ssml.encode("utf-8"))
            if resp.status_code == 401:
                logger.error("Azure TTS: invalid key (401) — check AZURE_TTS_KEY",
                             extra={"session_id": self.session_id})
                return False
            if resp.status_code == 400:
                logger.error("Azure TTS: bad request (400): %s", resp.text[:300],
                             extra={"session_id": self.session_id})
                return False
            if resp.status_code != 200:
                logger.error(
                    "Azure TTS API error %d: %s",
                    resp.status_code, resp.text[:200],
                    extra={"session_id": self.session_id},
                )
                return False
            mp3_bytes = resp.content

            if not mp3_bytes:
                return False

            loop = asyncio.get_running_loop()
            pcm_bytes = await loop.run_in_executor(
                None, lambda: _mp3_bytes_to_pcm(mp3_bytes, denoise=False)
            )
            if pcm_bytes is None:
                return False

            _debug_dump_audio_pair(
                provider="azure_tts",
                session_id=self.session_id,
                source_audio=mp3_bytes,
                source_ext="mp3",
                decoded_pcm=pcm_bytes,
            )
            logger.info(
                "Azure TTS success: voice=%s region=%s (%d bytes)",
                voice_name, region, len(mp3_bytes),
                extra={"session_id": self.session_id},
            )
            return await self._stream_pcm(pcm_bytes)

        except Exception as exc:
            logger.error("Azure TTS error: %s", exc,
                         extra={"session_id": self.session_id})
            return False

    # ------------------------------------------------------------------
    # Amazon Polly TTS
    # Docs: https://docs.aws.amazon.com/polly/latest/dg/API_SynthesizeSpeech.html
    # Requires: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME
    # Voice: AWS_POLLY_VOICE_TELUGU (default: Aditi/hi-IN — Telugu not natively in Polly)
    # Override AWS_POLLY_LANGUAGE_TELUGU if/when Polly adds a native te-IN voice.
    # ------------------------------------------------------------------

    async def _synthesize_amazon_polly(self, text: str) -> bool:
        if self._cancel_event.is_set():
            return False

        access_key = config.tts.amazon_polly_access_key
        secret_key = config.tts.amazon_polly_secret_key
        region     = config.tts.amazon_polly_region

        if not access_key or not secret_key:
            logger.warning(
                "Amazon Polly: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set, skipping",
                extra={"session_id": self.session_id},
            )
            return False

        voice_id      = (
            self._lang_cfg.get("amazon_polly_voice", "Aditi")
            if self._voice == "female"
            else self._lang_cfg.get("amazon_polly_voice_male", "Aditi")
        )
        language_code = self._lang_cfg.get("amazon_polly_language_code", "hi-IN")
        engine        = self._lang_cfg.get("amazon_polly_engine", "standard")

        try:
            import boto3
        except ImportError:
            logger.warning(
                "Amazon Polly: boto3 not installed — pip install boto3",
                extra={"session_id": self.session_id},
            )
            return False

        try:
            loop = asyncio.get_running_loop()

            def _call_polly() -> bytes:
                client = boto3.client(
                    "polly",
                    region_name=region,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                )
                resp = client.synthesize_speech(
                    Text=text,
                    OutputFormat="mp3",
                    VoiceId=voice_id,
                    LanguageCode=language_code,
                    Engine=engine,
                )
                return resp["AudioStream"].read()

            mp3_bytes = await loop.run_in_executor(None, _call_polly)

            if not mp3_bytes:
                return False

            pcm_bytes = await loop.run_in_executor(
                None, lambda: _mp3_bytes_to_pcm(mp3_bytes, denoise=False)
            )
            if pcm_bytes is None:
                return False

            _debug_dump_audio_pair(
                provider="amazon_polly",
                session_id=self.session_id,
                source_audio=mp3_bytes,
                source_ext="mp3",
                decoded_pcm=pcm_bytes,
            )
            logger.info(
                "Amazon Polly TTS success: voice=%s engine=%s (%d bytes)",
                voice_id, engine, len(mp3_bytes),
                extra={"session_id": self.session_id},
            )
            return await self._stream_pcm(pcm_bytes)

        except Exception as exc:
            logger.error("Amazon Polly TTS error: %s", exc,
                         extra={"session_id": self.session_id})
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
            communicate = edge_tts.Communicate(text, voice=self._edge_tts_voice, rate=config.tts.tts_rate)
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
        capture = _pcm_capture_var.get()
        for _ in text.split():
            if self._cancel_event.is_set():
                return False
            if capture is not None:
                capture.append(silence)
            else:
                await self._send_audio(silence)
                await asyncio.sleep(0.2)
        return True

    # ------------------------------------------------------------------
    # Common PCM streamer
    # ------------------------------------------------------------------

    async def _stream_pcm(self, pcm_bytes: bytes) -> bool:
        """Send PCM bytes — streams to client, or captures into task-local buffer.

        When _pcm_capture_var is set in the calling task's context (i.e. called
        from synthesize_to_pcm), appends the full pcm_bytes to that buffer
        instead of streaming. This is safe for concurrent use because asyncio
        task contexts are isolated.
        """
        capture = _pcm_capture_var.get()
        if capture is not None:
            capture.append(pcm_bytes)
            return not self._cancel_event.is_set()
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

    # Flush on sentence boundaries only.
    # Avoid comma-based flushes because they over-segment responses and increase
    # cloud TTS round-trips (higher latency, choppier playback).
    _SENTENCE_END = frozenset(".!?।")
    _MIN_FLUSH_CHARS = 1
    # Safety net: force-flush quickly for non-streaming cloud TTS providers.
    # Sarvam returns a complete audio file, so smaller first chunks reduce
    # time-to-first-audio while the next chunk synthesizes in parallel.
    _MAX_BUFFER_CHARS = 30

    async def run(self) -> None:
        """3-stage pipelined TTS loop.

        Stage 1 (_accumulate): reads LLM token fragments, assembles sentences,
          emits to sentence_queue.
        Stage 2 (_synthesize): reads sentences, calls synthesize_to_pcm (which
          runs in its own task context so _pcm_capture_var is isolated), emits
          PCM bytes to pcm_queue.
        Stage 3 (_stream): reads PCM bytes, streams to client in chunks.

        Synthesis of sentence N+1 overlaps with streaming of sentence N,
        cutting per-sentence latency by ~1s with Azure TTS / ~350ms with edge.
        """
        self._active = True
        logger.debug("TTSOrchestrator started", extra={"session_id": self.session_id})

        sentence_queue: asyncio.Queue = asyncio.Queue()
        pcm_queue: asyncio.Queue = asyncio.Queue()

        def _drain_fragments() -> None:
            while not self._fragment_queue.empty():
                try:
                    self._fragment_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        async def _accumulate() -> None:
            buf: list[str] = []
            try:
                while True:
                    if self._cancel_event.is_set():
                        _drain_fragments()
                        break
                    try:
                        fragment = await asyncio.wait_for(
                            self._fragment_queue.get(), timeout=0.1
                        )
                    except asyncio.TimeoutError:
                        continue

                    if fragment is None:
                        if buf and not self._cancel_event.is_set():
                            sentence_queue.put_nowait(" ".join(buf).strip())
                        break

                    if self._cancel_event.is_set():
                        _drain_fragments()
                        break

                    buf.append(fragment)
                    buf_text = " ".join(buf)
                    ends = (
                        fragment.rstrip()
                        and fragment.rstrip()[-1] in self._SENTENCE_END
                    )
                    force = len(buf_text) >= self._MAX_BUFFER_CHARS
                    if (ends or force) and len(buf_text) >= self._MIN_FLUSH_CHARS:
                        sentence_queue.put_nowait(buf_text)
                        buf = []
            finally:
                sentence_queue.put_nowait(None)

        async def _synthesize() -> None:
            try:
                while True:
                    text = await sentence_queue.get()
                    if text is None or self._cancel_event.is_set():
                        break
                    pcm = await self._tts.synthesize_to_pcm(text)
                    pcm_queue.put_nowait(pcm if pcm else b"")
            finally:
                pcm_queue.put_nowait(None)

        async def _stream() -> None:
            bpc = int(TTS_RATE * config.tts.chunk_ms / 1000) * 2
            while True:
                pcm = await pcm_queue.get()
                if pcm is None:
                    break
                if not pcm or self._cancel_event.is_set():
                    continue
                for i in range(0, len(pcm), bpc):
                    if self._cancel_event.is_set():
                        break
                    chunk = pcm[i: i + bpc]
                    await self._tts._send_audio(chunk)
                    self._tts.last_pcm_bytes_sent += len(chunk)
                    await asyncio.sleep(0)

        try:
            await asyncio.gather(
                asyncio.create_task(_accumulate(), name=f"tts-acc-{self.session_id[:8]}"),
                asyncio.create_task(_synthesize(), name=f"tts-syn-{self.session_id[:8]}"),
                asyncio.create_task(_stream(),     name=f"tts-str-{self.session_id[:8]}"),
            )
        except asyncio.CancelledError:
            raise
        finally:
            self._active = False
            logger.debug("TTSOrchestrator stopped", extra={"session_id": self.session_id})

    def is_active(self) -> bool:
        return self._active
