"""
config.py — Centralized configuration for the Telugu Voice AI Agent.

All tunable parameters can be overridden via environment variables.
No external API keys required for LLM inference — runs entirely on local GPU via Ollama.
"""

import os
from dataclasses import dataclass, field


@dataclass
class SonioxConfig:
    """Soniox streaming ASR — best-in-class Telugu speech recognition."""

    @property
    def api_key(self) -> str:
        return os.getenv("SONIOX_API_KEY", "")

    # Telugu language code
    language_code: str = os.getenv("SONIOX_LANGUAGE", "te")

    # Multilingual model that supports Telugu
    model: str = os.getenv("SONIOX_MODEL", "soniox_multilingual_2")

    # Audio format: PCM 16-bit signed little-endian from the browser
    audio_format: str = "pcm_s16le"
    sample_rate_hertz: int = 16000
    num_audio_channels: int = 1
    include_nonfinal: bool = True   # stream partial results for real-time display


@dataclass
class OllamaConfig:
    """
    Ollama local LLM — open-source, no API key required.

    Recommended Telugu models (pick one based on available VRAM):
      qwen2.5:32b   → ~20 GB VRAM  ← best Telugu quality
      qwen2.5:14b   → ~10 GB VRAM  ← good balance
      qwen2.5:7b    →  ~6 GB VRAM  ← runs on smaller GPUs
      gemma3:27b    → ~18 GB VRAM  ← Google Gemma 3
      aya:35b       → ~22 GB VRAM  ← Cohere multilingual

    Pull the model before starting:
      ollama pull qwen2.5:32b
    """
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:32b")
    temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
    top_p: float = float(os.getenv("OLLAMA_TOP_P", "0.9"))
    max_tokens: int = int(os.getenv("OLLAMA_MAX_TOKENS", "150"))
    # Sentence-boundary tokens that flush a TTS fragment
    sentence_delimiters: tuple = (".", "!", "?", ",", "।")


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


@dataclass
class MemoryConfig:
    """Sliding-window conversation history."""
    max_turns: int = int(os.getenv("MEMORY_MAX_TURNS", "8"))


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

    # TTS output: edge-tts native rate after PyAV resampling
    # MUST match PLAYBACK_SAMPLE_RATE in frontend/index.html
    tts_sample_rate: int = 24000

    # Pre-buffer before playback starts (ms)
    playback_prebuffer_ms: int = 200


@dataclass
class TTSConfig:
    """Text-to-Speech — Telugu neural voices via edge-tts (free, no API key)."""
    # Telugu voices:
    #   te-IN-ShrutiNeural  ← female (recommended)
    #   te-IN-MohanNeural   ← male
    edge_tts_voice: str = os.getenv("TTS_VOICE", "te-IN-ShrutiNeural")

    # gTTS fallback language code
    gtts_language: str = "te"

    # Output sample rate (target after PyAV resample — matches AudioConfig.tts_sample_rate)
    sample_rate: int = 24000

    # Stream in N-ms chunks for low cancel latency
    chunk_ms: int = 60


@dataclass
class ServerConfig:
    """FastAPI / uvicorn settings."""
    host: str = os.getenv("SERVER_HOST", "0.0.0.0")
    port: int = int(os.getenv("SERVER_PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "info")
    cors_origins: list = field(default_factory=lambda: ["*"])


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

    # System persona injected into every LLM prompt
    system_prompt: str = (
        "మీరు Qobox (Quality Outside The Box) కోసం ఒక స్నేహపూర్వక మరియు వృత్తిపరమైన "
        "కస్టమర్ సర్వీస్ వాయిస్ ఏజెంట్. Qobox అనేది భారతీయ సాఫ్ట్‌వేర్ నాణ్యత నిర్ధారణ "
        "మరియు పరీక్ష సేవల సంస్థ. మీరు లైవ్ ఫోన్ కాల్‌లో ఉన్నారు. "
        "కస్టమర్ తెలుగులో మాట్లాడితే తెలుగులో సమాధానం ఇవ్వండి. "
        "కస్టమర్ ఇంగ్లీష్‌లో మాట్లాడితే ఇంగ్లీష్‌లో సమాధానం ఇవ్వండి. "
        "ప్రతి సమాధానం గరిష్టంగా 1-2 చిన్న వాక్యాలుగా ఉంచండి — ఇది వాయిస్ కాల్. "
        "bullet points, markdown, asterisks లేదా emojis వాడకండి. "
        "సంభాషణ చరిత్రను జాగ్రత్తగా చదవండి. వినియోగదారు పేరు చెప్పినట్లయితే, దాన్ని వాడండి. "
        "Qobox సేవలకు (సాఫ్ట్‌వేర్ టెస్టింగ్, ఆటోమేషన్, పెర్ఫార్మెన్స్ టెస్టింగ్, "
        "సెక్యూరిటీ టెస్టింగ్, API టెస్టింగ్, QA కన్సల్టింగ్) సంబంధించిన ప్రశ్నలకు మాత్రమే సమాధానం ఇవ్వండి. "
        "Qobox కి సంబంధం లేని విషయాలకు: "
        "'క్షమించండి, నేను Qobox విషయాలలో మాత్రమే సహాయం చేయగలను' అని చెప్పండి. "
        "తెలియని విషయాలకు: 'అందుకు నేను మీకు నిపుణులకు కనెక్ట్ చేస్తాను' అని చెప్పండి.\n\n"
        "ENGLISH FALLBACK: If the customer speaks English, respond naturally in English. "
        "You are a friendly Qobox assistant. Keep responses to 1-2 short sentences. "
        "Only answer questions about Qobox software testing and QA services."
    )


# Module-level singleton — import this everywhere
config = AppConfig()
