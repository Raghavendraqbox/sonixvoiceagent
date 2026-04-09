"""
config.py — Centralized configuration for the Dari & Pashto Voice AI Agent.

All tunable parameters can be overridden via environment variables.
Language is selected via LANGUAGE env var (dari | pashto) or per-session via
the WebSocket query parameter ?language=dari / ?language=pashto.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple


# ---------------------------------------------------------------------------
# Language-specific configurations
# ---------------------------------------------------------------------------

LANGUAGE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "dari": {
        # Display
        "display_name": "Dari",
        "display_name_native": "دری",

        # ASR
        "soniox_language_code": "fa",    # Soniox: Persian covers Dari
        "whisper_language": "fa",         # faster-whisper language code

        # TTS — Meta MMS-TTS (primary, local GPU)
        "mms_tts_model": "facebook/mms-tts-fas",   # Afghan Persian / Dari (ISO 639-3: fas)
        "mms_tts_sample_rate": 16_000,

        # TTS — edge-tts (fallback 1, Microsoft Azure)
        "edge_tts_voice": os.getenv("TTS_VOICE_DARI", "fa-IR-DilaraNeural"),   # female
        "edge_tts_voice_male": "fa-IR-FaridNeural",

        # TTS — gTTS (fallback 2, Google)
        "gtts_language": "fa",

        # LLM sentence boundaries (Arabic punctuation added)
        "sentence_delimiters": (".", "!", "?", ",", "؟", "،", "۔"),

        # Greeting played on first user utterance
        "greeting": (
            "به Qobox خوش آمدید. من دستیار مجازی شما هستم. "
            "چطور می‌توانم کمکتان کنم؟"
        ),

        # Stubs when Ollama is unreachable
        "neutral_stubs": [
            "ببخشید، لطفاً یک لحظه صبر کنید.",
            "متوجه شدم، اجازه دهید بررسی کنم.",
            "بسیار ممنون، یک لحظه صبر کنید.",
        ],

        # System persona injected into every LLM prompt
        "system_prompt": (
            "You are a friendly and professional customer service voice agent for "
            "Qobox (Quality Outside The Box), an Indian software quality assurance "
            "and testing services company. You are on a live phone call.\n\n"
            "IMPORTANT: Always respond in Dari (Afghan Persian / دری). Write your "
            "response in Persian/Dari script only. Never respond in English or any "
            "other language — even if the transcription appears in English, always "
            "reply in Dari. Keep each response to 1-2 short sentences — this "
            "is a voice call, so be conversational and concise. Do not use bullet "
            "points, markdown formatting, asterisks, or emojis. Read the "
            "conversation history carefully. If the user mentions their name, use it.\n\n"
            "Only answer questions about Qobox services: software testing, test "
            "automation, performance testing, security testing, API testing, and "
            "QA consulting. For off-topic questions, say: "
            "'متأسفم، من فقط می‌توانم در مورد خدمات Qobox کمک کنم.' "
            "For unknown questions, say: "
            "'اجازه دهید شما را به یک متخصص وصل کنم.'"
        ),
    },

    "pashto": {
        # Display
        "display_name": "Pashto",
        "display_name_native": "پښتو",

        # ASR
        "soniox_language_code": "ps",    # Pashto — test via Soniox API; may fall back
        "whisper_language": "ps",         # faster-whisper supports Pashto

        # TTS — Meta MMS-TTS (primary, local GPU — same engine as Dari)
        "mms_tts_model": "facebook/mms-tts-pbt",   # Southern Pashto (most common in AF)
        "mms_tts_sample_rate": 16_000,

        # TTS — edge-tts (fallback 1, Microsoft Azure)
        "edge_tts_voice": os.getenv("TTS_VOICE_PASHTO", "ps-AF-LatifaNeural"),
        "edge_tts_voice_male": "ps-AF-GulNawazNeural",

        # TTS — gTTS (fallback 2, Google — Pashto not supported; use Persian as closest)
        "gtts_language": "fa",

        # LLM sentence boundaries (Arabic punctuation)
        "sentence_delimiters": (".", "!", "?", ",", "؟", "،"),

        # Greeting played on first user utterance
        "greeting": (
            "Qobox ته ښه راغلاست. زه ستاسو مجازی مرستیال یم. "
            "زه تاسو سره څنګه مرسته کولی شم؟"
        ),

        # Stubs when Ollama is unreachable
        "neutral_stubs": [
            "بخښنه وغواړئ، یو شیبه صبر وکړئ.",
            "پوه شوم، اجازه راکړئ وګورم.",
            "مننه، یو شیبه صبر وکړئ.",
        ],

        # System persona injected into every LLM prompt
        "system_prompt": (
            "You are a friendly and professional customer service voice agent for "
            "Qobox (Quality Outside The Box), an Indian software quality assurance "
            "and testing services company. You are on a live phone call.\n\n"
            "IMPORTANT: Always respond in Pashto (پښتو). Write your response in "
            "Pashto script only. Never respond in English or any other language — "
            "even if the transcription appears in English, always reply in Pashto. "
            "Keep each response to 1-2 short sentences — this is a voice call, so "
            "be conversational and concise. Do not use bullet points, markdown "
            "formatting, asterisks, or emojis. Read the conversation history "
            "carefully. If the user mentions their name, use it.\n\n"
            "Only answer questions about Qobox services: software testing, test "
            "automation, performance testing, security testing, API testing, and "
            "QA consulting. For off-topic questions, say: "
            "'بخښنه غواړم، زه یوازې د Qobox خدماتو په اړه مرسته کولی شم.' "
            "For unknown questions, say: "
            "'اجازه راکړئ چې تاسو یو متخصص سره وصل کړم.'"
        ),
    },
}

SUPPORTED_LANGUAGES = list(LANGUAGE_CONFIGS.keys())
DEFAULT_LANGUAGE: str = os.getenv("LANGUAGE", "dari").lower()

if DEFAULT_LANGUAGE not in SUPPORTED_LANGUAGES:
    DEFAULT_LANGUAGE = "dari"


def get_language_config(language: str) -> Dict[str, Any]:
    """Return the config dict for the given language (defaults to Dari)."""
    return LANGUAGE_CONFIGS.get(language.lower(), LANGUAGE_CONFIGS["dari"])


# ---------------------------------------------------------------------------
# Soniox streaming ASR config
# ---------------------------------------------------------------------------

@dataclass
class SonioxConfig:
    """Soniox streaming ASR — cloud-based, supports Dari (fa) and Pashto (ps)."""

    @property
    def api_key(self) -> str:
        return os.getenv("SONIOX_API_KEY", "")

    # Model: stt-rt-v4 is the current recommended real-time model (soniox_multilingual_2 is legacy)
    model: str = os.getenv("SONIOX_MODEL", "stt-rt-v4")

    # Audio format: PCM 16-bit signed little-endian from the browser
    audio_format: str = "pcm_s16le"
    sample_rate_hertz: int = 16000
    num_audio_channels: int = 1
    include_nonfinal: bool = True   # stream partial results for real-time display


# ---------------------------------------------------------------------------
# Ollama LLM config
# ---------------------------------------------------------------------------

@dataclass
class OllamaConfig:
    """
    Ollama local LLM — open-source, no API key required.

    Recommended models for Dari/Pashto (pull before starting):
      qwen2.5:7b    → ~6 GB VRAM   ← good Dari, reasonable Pashto
      qwen2.5:14b   → ~10 GB VRAM  ← better multilingual quality
      qwen2.5:32b   → ~20 GB VRAM  ← best Dari quality
      qwen2.5:72b   → ~48 GB VRAM  ← best overall (default for 80GB GPU)
      aya-expanse   → varies       ← Cohere multilingual, Persian supported

    For Pashto-specific fine-tuned model:
      junaid008/qehwa-pashto-llm  ← Convert GGUF to Ollama Modelfile

    Pull the model before starting:
      ollama pull qwen2.5:7b
    """
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
    top_p: float = float(os.getenv("OLLAMA_TOP_P", "0.9"))
    max_tokens: int = int(os.getenv("OLLAMA_MAX_TOKENS", "150"))


# ---------------------------------------------------------------------------
# RAG config
# ---------------------------------------------------------------------------

@dataclass
class RAGConfig:
    """FAISS-backed RAG for Qobox knowledge base."""
    embedding_model: str = os.getenv(
        "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    docs_directory: str = os.getenv("RAG_DOCS_DIR", "./docs")
    index_path: str = os.getenv("RAG_INDEX_PATH", "./faiss_index")
    top_k: int = int(os.getenv("RAG_TOP_K", "3"))
    similarity_threshold: float = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.3"))
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "300"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))


# ---------------------------------------------------------------------------
# Memory config
# ---------------------------------------------------------------------------

@dataclass
class MemoryConfig:
    """Sliding-window conversation history."""
    max_turns: int = int(os.getenv("MEMORY_MAX_TURNS", "8"))


# ---------------------------------------------------------------------------
# Audio config
# ---------------------------------------------------------------------------

@dataclass
class AudioConfig:
    """Audio pipeline settings — must stay in sync with frontend."""
    # Microphone capture: 16kHz mono 16-bit → 3200 bytes per 100ms chunk
    input_chunk_bytes: int = 3200
    input_sample_rate: int = 16000
    input_channels: int = 1
    input_bit_depth: int = 16

    # VAD: RMS energy threshold for speech vs. silence
    vad_rms_threshold: float = float(os.getenv("VAD_RMS_THRESHOLD", "0.01"))

    # TTS output: resampled to 24kHz for browser playback
    # MUST match PLAYBACK_SAMPLE_RATE in frontend/index.html
    tts_sample_rate: int = 24000

    # Pre-buffer before playback starts (ms)
    playback_prebuffer_ms: int = 200


# ---------------------------------------------------------------------------
# TTS config
# ---------------------------------------------------------------------------

@dataclass
class TTSConfig:
    """Text-to-Speech output settings (language-specific voices set in LANGUAGE_CONFIGS)."""
    # Output sample rate (target after resample — matches AudioConfig.tts_sample_rate)
    sample_rate: int = 24000

    # Stream in N-ms chunks for low cancel latency
    chunk_ms: int = 60


# ---------------------------------------------------------------------------
# Server config
# ---------------------------------------------------------------------------

@dataclass
class ServerConfig:
    """FastAPI / uvicorn settings."""
    host: str = os.getenv("SERVER_HOST", "0.0.0.0")
    port: int = int(os.getenv("SERVER_PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "info")
    cors_origins: list = field(default_factory=lambda: ["*"])


# ---------------------------------------------------------------------------
# Root application config
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    """Root application config — single source of truth."""
    soniox: SonioxConfig = field(default_factory=SonioxConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    # Default language for the server
    default_language: str = DEFAULT_LANGUAGE


# Module-level singleton — import this everywhere
config = AppConfig()
