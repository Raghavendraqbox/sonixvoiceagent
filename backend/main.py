"""
main.py — FastAPI application with WebSocket endpoint for the Telugu/Kannada Voice AI agent.

Language selection:
  Pass ?language=telugu or ?language=kannada in the WebSocket URL.
  Defaults to the LANGUAGE environment variable (default: "telugu").

WebSocket message protocol:
  CLIENT → SERVER (binary):
    Raw PCM audio bytes (16-bit, mono, 16kHz, ~3200 bytes per 100ms chunk)

  CLIENT → SERVER (JSON text):
    {"type": "interrupt"}              — user VAD detected speech mid-response
    {"type": "ping"}                   — keepalive

  SERVER → CLIENT (binary):
    Raw PCM audio bytes (TTS output, 24000 Hz, 16-bit, mono)

  SERVER → CLIENT (JSON text):
    {"type": "tts_start"}             — first TTS chunk about to arrive
    {"type": "tts_end"}               — TTS fully delivered (or cancelled)
    {"type": "tts_stopped"}           — TTS interrupted mid-stream
    {"type": "transcript_partial", "text": "..."}  — ASR interim result
    {"type": "transcript_final",  "text": "..."}   — ASR final result
    {"type": "bot_text_fragment",  "text": "..."}  — streamed LLM token group
    {"type": "error", "message": "..."}            — pipeline error
    {"type": "pong"}                               — keepalive reply
    {"type": "session_ready", "session_id": "...", "language": "..."}
"""

import asyncio
import json
import logging
import math
import os
import struct
import sys
from contextlib import asynccontextmanager

# Load .env from repo root (one level above backend/)
from pathlib import Path
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path, override=True)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import (
    SARVAM_EMOTION_TEMPERATURES,
    SARVAM_FEMALE_SPEAKERS,
    BUSINESS_CONFIGS,
    config,
    SUPPORTED_BUSINESSES,
    SUPPORTED_LANGUAGES,
)
from session_manager import session_manager

# Windows terminals often default to cp1252, which cannot print Telugu/Kannada.
# Reconfigure stdout so logging those transcripts does not block the event loop
# with repeated UnicodeEncodeError tracebacks.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    stream=sys.stdout,
    level=getattr(logging, config.server.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _pcm_rms(pcm_bytes: bytes) -> float:
    """Return normalized RMS for 16-bit little-endian mono PCM."""
    sample_count = len(pcm_bytes) // 2
    if sample_count <= 0:
        return 0.0
    total = 0
    for (sample,) in struct.iter_unpack("<h", pcm_bytes[: sample_count * 2]):
        total += sample * sample
    return math.sqrt(total / sample_count) / 32768.0


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Telugu & Kannada Voice AI Agent starting…")
    logger.info("Default language : %s", config.default_language)
    logger.info("Supported        : %s", ", ".join(SUPPORTED_LANGUAGES))
    logger.info("STT  : %s (default engine: %s)", config.soniox.model, config.default_stt_engine)
    logger.info("LLM  : Ollama %s @ %s", config.ollama.model, config.ollama.base_url)
    logger.info("TTS  : Telugu=%s | Kannada=%s", config.tts.telugu_engine_priority, config.tts.kannada_engine_priority)
    logger.info("Audio: input 16kHz | TTS output 24kHz")
    logger.info("=" * 60)
    session_manager.initialize_rag()
    await session_manager.warmup_llm()
    from tts import warmup_tts_connection
    await warmup_tts_connection()
    logger.info("Server ready.")
    yield
    logger.info("Shutting down…")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Telugu & Kannada Voice AI Agent",
    description=(
        "Real-time full-duplex Telugu and Kannada conversational voice agent. "
        "Powered by multi-provider ASR + Ollama LLM + multi-provider TTS."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect browser to the frontend."""
    return HTMLResponse(
        content='<meta http-equiv="refresh" content="0; url=/static/index.html">',
        status_code=200,
    )


@app.get("/health")
async def health():
    """Health check — shows which ASR/LLM/TTS backend is active."""
    return {
        "status": "ok",
        "supported_languages": SUPPORTED_LANGUAGES,
        "default_language": config.default_language,
        "asr": (
            f"soniox/{config.soniox.model}"
            if os.getenv("SONIOX_API_KEY")
            else "sarvam-stt/saarika:v2.5"
            if os.getenv("SARVAM_API_KEY")
            else "whisper-large-v3 (local GPU)"
        ),
        "llm": (
            f"gemini/{config.gemini.model}"
            if config.default_llm_backend == "gemini" and config.gemini.api_key
            else f"ollama/{config.ollama.model} @ {config.ollama.base_url}"
        ),
        "tts": (
            f"telugu: {config.tts.telugu_engine_priority} | "
            f"kannada: {config.tts.kannada_engine_priority}"
        ),
    }


@app.get("/languages")
async def languages():
    """Return available language options for the frontend."""
    from config import LANGUAGE_CONFIGS
    return {
        lang: {
            "display_name": cfg["display_name"],
            "display_name_native": cfg["display_name_native"],
        }
        for lang, cfg in LANGUAGE_CONFIGS.items()
    }


@app.get("/client-config")
async def client_config():
    """Return non-secret browser runtime settings derived from .env."""
    return {
        "audio": {
            "office_background_noise_enabled": config.audio.office_background_noise_enabled,
            "keyboard_typing_sound_enabled": config.audio.keyboard_typing_sound_enabled,
            "office_background_noise_gain": config.audio.office_background_noise_gain,
            "keyboard_typing_sound_gain": config.audio.keyboard_typing_sound_gain,
            "keyboard_typing_min_ms": config.audio.keyboard_typing_min_ms,
            "keyboard_typing_max_ms": config.audio.keyboard_typing_max_ms,
        },
        "sarvam": {
            "female_speakers": list(SARVAM_FEMALE_SPEAKERS),
            "emotions": list(SARVAM_EMOTION_TEMPERATURES.keys()),
            "default_emotion": config.tts.sarvam_emotion,
            "default_pace": config.tts.sarvam_pace,
            "min_pace": config.tts.sarvam_pace_min,
            "max_pace": config.tts.sarvam_pace_max,
            "pace_step": config.tts.sarvam_pace_step,
        },
        "businesses": {
            business: {
                "display_name": cfg["display_name"],
                "description": cfg.get("description", ""),
            }
            for business, cfg in BUSINESS_CONFIGS.items()
        },
    }


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    language: str = "telugu",
    business: str = "mercotrace",
    voice: str = "male",
    tts_engine: str = "auto",
    sarvam_speaker: str = "",
    sarvam_emotion: str = "",
    sarvam_pace: float = 0.0,
    stt_engine: str = "",
    llm_backend: str = "",
):
    """
    Main WebSocket handler.

    Query parameters:
      language   — "telugu" or "kannada" (defaults to LANGUAGE env var → "telugu")
      business   — "mercotrace" or "davia_hospital"
      voice      — "male" (default) or "female"
      sarvam_speaker — female Sarvam speaker override, e.g. "anushka"
      sarvam_emotion — Bulbul v3 emotion preset: neutral | calm | warm | empathetic |
                       happy | cheerful | excited | serious
      tts_engine — Kannada TTS engine override: auto | elevenlabs | mms | edge |
                   narakeet | micmonster | speakatoo | gtts
                   "auto" uses KANNADA_TTS_ENGINE_PRIORITY from .env
      stt_engine — STT engine: auto | sarvam | soniox | whisper
                   "auto" follows the priority chain (Sarvam → Soniox → Whisper)
    """
    # Normalize and validate
    language = language.lower()
    if language not in SUPPORTED_LANGUAGES:
        language = config.default_language
    business = business.lower().strip()
    if business not in SUPPORTED_BUSINESSES:
        business = config.default_business
    voice = voice.lower() if voice.lower() in ("male", "female") else "male"
    tts_engine = tts_engine.lower().strip()
    sarvam_speaker = sarvam_speaker.lower().strip()
    if sarvam_speaker not in SARVAM_FEMALE_SPEAKERS:
        sarvam_speaker = ""
    sarvam_emotion = sarvam_emotion.lower().strip() or config.tts.sarvam_emotion
    if sarvam_emotion not in SARVAM_EMOTION_TEMPERATURES:
        sarvam_emotion = "neutral"
    # Clamp the optional Sarvam pace slider value to the API's supported range.
    # 0.0 = "use server default" (preserves the legacy SARVAM_PACE behaviour).
    try:
        sarvam_pace = float(sarvam_pace)
    except (TypeError, ValueError):
        sarvam_pace = 0.0
    if sarvam_pace > 0:
        sarvam_pace = max(0.5, min(2.0, sarvam_pace))
    else:
        sarvam_pace = 0.0
    stt_engine = stt_engine.lower().strip() or config.default_stt_engine
    if stt_engine not in ("auto", "sarvam", "soniox", "google", "azure", "amazon", "whisper"):
        stt_engine = "auto"
    llm_backend = llm_backend.lower().strip() or config.default_llm_backend
    if llm_backend not in ("ollama", "gemini"):
        llm_backend = "ollama"

    await websocket.accept()
    logger.info(
        "WebSocket connected from %s (language=%s, business=%s, voice=%s, tts=%s, "
        "sarvam_speaker=%s, sarvam_emotion=%s, sarvam_pace=%s, stt=%s, llm=%s)",
        websocket.client,
        language,
        business,
        voice,
        tts_engine,
        sarvam_speaker or "default",
        sarvam_emotion,
        f"{sarvam_pace:.2f}" if sarvam_pace else "default",
        stt_engine,
        llm_backend,
    )

    async def send_audio(pcm_bytes: bytes) -> None:
        try:
            # Mark bot audio active only when the first PCM chunk is actually sent.
            # This prevents pre-audio interrupt messages from cancelling the turn.
            session.bot_audio_active = True
            await websocket.send_bytes(pcm_bytes)
        except Exception as exc:
            logger.debug("send_audio error: %s", exc)

    async def send_json_msg(payload: dict) -> None:
        try:
            await websocket.send_text(json.dumps(payload))
        except Exception as exc:
            logger.debug("send_json error: %s", exc)

    session = session_manager.create_session(
        send_audio_cb=send_audio,
        send_json_cb=send_json_msg,
        language=language,
        business=business,
        voice=voice,
        tts_engine=tts_engine,
        sarvam_speaker=sarvam_speaker,
        sarvam_emotion=sarvam_emotion,
        sarvam_pace=sarvam_pace,
        stt_engine=stt_engine,
        llm_backend=llm_backend,
    )
    await send_json_msg({
        "type": "session_ready",
        "session_id": session.session_id,
        "language": language,
        "business": business,
    })
    logger.info("Session %s ready [%s]", session.session_id, language)

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect

            # Binary frame — raw PCM audio from microphone
            if "bytes" in message and message["bytes"]:
                pcm_chunk: bytes = message["bytes"]
                if len(pcm_chunk) % 2 != 0:
                    await send_json_msg(
                        {"type": "error", "message": "PCM chunk has odd byte count"}
                    )
                    continue
                # Silero (or RMS-fallback) VAD classifies the chunk as speech /
                # non-speech BEFORE it reaches barge-in logic or the STT engine.
                # Background TV / chatter / fan noise gets tagged is_speech=False
                # so neither the interrupt path nor the STT utterance buffer
                # picks it up.
                if session.vad is not None:
                    is_speech_raw = bool(session.vad.is_speech(pcm_chunk))
                else:
                    is_speech_raw = (
                        _pcm_rms(pcm_chunk) > config.audio.vad_rms_threshold
                    )

                # Echo guard: while the bot is actively playing TTS, the
                # microphone almost always picks up the bot's own audio
                # bleeding through the speakers (browser AEC is imperfect on
                # laptops/phones). Tagging those frames as is_speech=True
                # for the STT engine causes it to buffer and transcribe the
                # bot's greeting back as user input — exactly the "welcome"
                # loop seen in production logs. We keep the raw VAD verdict
                # for barge-in detection (separate counter below), but
                # suppress the STT-facing tag during bot playback. Once the
                # user genuinely barges in, cancel_tts() clears
                # bot_audio_active and normal flow resumes.
                is_speech_for_stt = is_speech_raw and not session.bot_audio_active

                if is_speech_raw:
                    session.user_speaking_event.set()
                    session.input_silence_frames = 0
                    if session.bot_audio_active:
                        session.bot_bargein_speech_frames += 1
                    else:
                        session.bot_bargein_speech_frames = 0
                    if (
                        session.tts_orchestrator
                        and session.tts_orchestrator.is_active()
                        and session.bot_audio_active
                        and session.bot_bargein_speech_frames >= 2
                        and not session.tts_cancel_event.is_set()
                    ):
                        session.cancel_tts()
                        await send_json_msg({"type": "tts_stopped"})
                elif session.user_speaking_event.is_set():
                    session.input_silence_frames += 1
                    session.bot_bargein_speech_frames = 0
                    # Browser sends ~100 ms chunks; wait for sustained silence
                    # before allowing the LLM to answer.
                    if session.input_silence_frames >= 3:
                        session.user_speaking_event.clear()
                else:
                    session.bot_bargein_speech_frames = 0
                # Push (bytes, is_speech_for_stt) so STT engines can rely on
                # the gated VAD verdict — they will neither buffer echoed
                # bot audio nor re-run an energy check.
                await session.audio_queue.put((pcm_chunk, is_speech_for_stt))

            # Text frame — JSON control message
            elif "text" in message and message["text"]:
                try:
                    msg = json.loads(message["text"])
                except json.JSONDecodeError:
                    logger.warning("Malformed JSON from client: %s", message["text"][:80])
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "interrupt":
                    if (
                        session.tts_orchestrator
                        and session.tts_orchestrator.is_active()
                        and session.bot_audio_active
                        and not session.tts_cancel_event.is_set()
                    ):
                        session.cancel_tts()
                        await send_json_msg({"type": "tts_stopped"})
                    else:
                        logger.debug(
                            "Ignoring interrupt before bot audio start",
                            extra={"session_id": session.session_id},
                        )

                elif msg_type == "transcript_partial":
                    session.interrupt_event.set()

                elif msg_type == "set_stt_engine":
                    new_engine = msg.get("engine", "auto")
                    session_manager.switch_asr_engine(session.session_id, new_engine)
                    await send_json_msg({"type": "stt_engine_changed", "engine": new_engine})
                    logger.info(
                        "STT engine switched to '%s' for session %s",
                        new_engine, session.session_id,
                    )

                elif msg_type == "ping":
                    await send_json_msg({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected (session %s)", session.session_id)
    except Exception as exc:
        logger.error("WebSocket error for session %s: %s", session.session_id, exc)
    finally:
        await session_manager.destroy_session(session.session_id)
        logger.info("Session %s destroyed", session.session_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level,
        workers=1,   # must be 1 — asyncio state is not fork-safe
    )
