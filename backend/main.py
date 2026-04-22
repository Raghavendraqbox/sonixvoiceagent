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
import os
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

from config import config, SUPPORTED_LANGUAGES
from session_manager import session_manager

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    stream=sys.stdout,
    level=getattr(logging, config.server.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Telugu & Kannada Voice AI Agent starting…")
    logger.info("Default language : %s", config.default_language)
    logger.info("Supported        : %s", ", ".join(SUPPORTED_LANGUAGES))
    logger.info("ASR  : Soniox (%s) → Whisper large-v3 fallback", config.soniox.model)
    logger.info("LLM  : Ollama %s @ %s", config.ollama.model, config.ollama.base_url)
    logger.info("TTS  : Telugu=ElevenLabs→MMS(tel) fallback | Kannada=ElevenLabs→edge→gTTS")
    logger.info("Audio: input 16kHz | TTS output 24kHz")
    logger.info("=" * 60)
    session_manager.initialize_rag()
    await session_manager.warmup_llm()
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
        "Powered by Soniox ASR + Ollama LLM + Meta MMS-TTS."
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
            else "whisper-large-v3 (local GPU)"
        ),
        "llm": f"ollama/{config.ollama.model} @ {config.ollama.base_url}",
        "tts": "telugu: mms-tts strict | kannada: configurable chain (24kHz output)",
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


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    language: str = "telugu",
    voice: str = "male",
    tts_engine: str = "auto",
    stt_engine: str = "auto",
):
    """
    Main WebSocket handler.

    Query parameters:
      language   — "telugu" or "kannada" (defaults to LANGUAGE env var → "telugu")
      voice      — "male" (default) or "female"
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
    voice = voice.lower() if voice.lower() in ("male", "female") else "male"
    tts_engine = tts_engine.lower().strip()
    stt_engine = stt_engine.lower().strip()
    if stt_engine not in ("auto", "sarvam", "soniox", "google", "azure", "amazon", "whisper"):
        stt_engine = "auto"

    await websocket.accept()
    logger.info(
        "WebSocket connected from %s (language=%s, voice=%s, tts_engine=%s, stt_engine=%s)",
        websocket.client,
        language,
        voice,
        tts_engine,
        stt_engine,
    )

    async def send_audio(pcm_bytes: bytes) -> None:
        try:
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
        voice=voice,
        tts_engine=tts_engine,
        stt_engine=stt_engine,
    )
    await send_json_msg({
        "type": "session_ready",
        "session_id": session.session_id,
        "language": language,
    })
    logger.info("Session %s ready [%s]", session.session_id, language)

    try:
        while True:
            message = await websocket.receive()

            # Binary frame — raw PCM audio from microphone
            if "bytes" in message and message["bytes"]:
                pcm_chunk: bytes = message["bytes"]
                if len(pcm_chunk) % 2 != 0:
                    await send_json_msg(
                        {"type": "error", "message": "PCM chunk has odd byte count"}
                    )
                    continue
                await session.audio_queue.put(pcm_chunk)

            # Text frame — JSON control message
            elif "text" in message and message["text"]:
                try:
                    msg = json.loads(message["text"])
                except json.JSONDecodeError:
                    logger.warning("Malformed JSON from client: %s", message["text"][:80])
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "interrupt":
                    session.cancel_tts()
                    await send_json_msg({"type": "tts_stopped"})

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
