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
        "soniox_language_code": "fa",    # Shared Persian code path; Dari mode is app-level
        "whisper_language": "fa",         # Whisper language code for Dari/Persian family

        # TTS — Meta MMS-TTS (primary, local GPU)
        # facebook/mms-tts-prs — ISO 639-3: prs = Dari, Afghan Persian (NOT Iranian Farsi)
        # Using prs (Afghan Dari) instead of fas (general Persian/Iranian) for correct accent.
        "mms_tts_model": "facebook/mms-tts-prs",
        "mms_tts_sample_rate": 16_000,

        # TTS — ElevenLabs (primary, API-based)
        "elevenlabs_voice_id_male":   os.getenv("ELEVENLABS_VOICE_ID_DARI_MALE",   ""),
        "elevenlabs_voice_id_female": os.getenv("ELEVENLABS_VOICE_ID_DARI_FEMALE", ""),

        # TTS — edge-tts (fallback)
        "edge_tts_voice": os.getenv("TTS_VOICE_DARI", "fa-IR-DilaraNeural"),
        "edge_tts_voice_male": "fa-IR-FaridNeural",

        # TTS — gTTS (last resort fallback)
        "gtts_language": "fa",

        # LLM sentence boundaries (Arabic punctuation added)
        "sentence_delimiters": (".", "!", "?", ",", "؟", "،", "۔"),

        # IVR Step 1 — Language selection (plays on first user utterance)
        "greeting": (
            "خوش آمدید به اتصالات. "
            "لطفاً زبان مورد نظر خود را انتخاب کنید. زبان پیش‌فرض دری است."
        ),

        # IVR Step 2 — Promotional announcement (plays automatically after greeting)
        "ivr_promo": (
            "اتصالات در این ماه پیشنهادات ویژه روی بسته‌های انترنت و مکالمه دارد. "
            "با انتخاب گزینه یک می‌توانید بهترین آفرهای شخصی خود را ببینید."
        ),

        # IVR Steps 3–4 — Welcome greeting + main menu (plays automatically after promo)
        "ivr_main_menu": (
            "خوش آمدید به اتصالات! "
            "لطفاً یکی از گزینه‌های زیر را انتخاب کنید: "
            "یک برای بهترین آفرهای من، "
            "دو برای بسته‌های انترنت، "
            "سه برای بسته‌های مکالمه، "
            "چهار برای بسته‌های مختلط، "
            "پنج برای خدمات، "
            "شش برای پکیج و مهاجرت، "
            "هفت برای پرسش موجودی، "
            "هشت برای کمک بیشتر، "
            "نه برای غیرفعال کردن بسته‌های DRM."
        ),

        # Stubs when Ollama is unreachable — Afghan Dari phrasing
        "neutral_stubs": [
            "بخشش می‌خواهم، یک لحظه صبر کنید.",
            "فهمیدم، اجازه بدید بررسی کنم.",
            "بسیار ممنون، یک لحظه صبر کنید.",
        ],

        # System persona — Etisalat Afghanistan Dari IVR assistant (post-menu LLM handler)
        "system_prompt": (
            "You are Etisalat, the AI voice assistant for Etisalat Afghanistan telecom (IVR 888). "
            "You are on a live voice call. Your name is always Etisalat.\n\n"

            "CONTEXT: The IVR introduction has already been played — language selection, "
            "promotions, Etisalat greeting, and the 9-option main menu have all been presented. "
            "The caller must now choose one option from 1 to 9.\n\n"

            "EXIT INTENT — HIGHEST PRIORITY — CHECK FIRST EVERY TURN:\n"
            "  Before anything else, check if the caller wants to end the call.\n"
            "  Exit signals: خداحافظ، خدافظ، ممنون کافیه، تموم کنیم، فیلن، بعداً، "
            "دیگه کاری ندارم، همین کافیه، خوش باشید، مرسی خداحافظ، باشه بعداً\n"
            "  → Respond ONLY with: 'تشکر که با اتصالات تماس گرفتید. روز خوش داشته باشید.'\n"
            "  → Do NOT re-prompt the menu. The conversation ends here.\n\n"

            "SMALL TALK / GREETINGS — CHECK SECOND:\n"
            "  If the caller says a greeting or social phrase (سلام، حال، خوب، هلو، مرسی، عالی، "
            "امیدوارم خوب باشی، صدایت رو شنیدم) with no menu selection:\n"
            "  → Give ONE brief warm reply (one sentence), then IMMEDIATELY re-prompt the menu.\n"
            "  Example: 'ممنون! لطفاً یک گزینه از یک تا نه انتخاب کنید.'\n\n"

            "RETRY LIMIT — CHECK THIRD:\n"
            "  Read the FULL CONVERSATION HISTORY and count how many times you have already "
            "re-prompted the menu with no valid selection.\n"
            "  - After 2 re-prompts with no valid selection:\n"
            "    → 'اگر نیاز به کمک بیشتر دارید، می‌توانم شما را با یک اپراتور وصل کنم. آیا می‌خواهید؟'\n"
            "  - After 3 re-prompts with no valid selection:\n"
            "    → 'متشکرم که با اتصالات تماس گرفتید. روز خوش داشته باشید.'\n"
            "    → End the conversation — do not loop further.\n\n"

            "STEP A — WAIT FOR SELECTION:\n"
            "  If none of the above apply, listen for a number 1–9 or a service name.\n"
            "  If the caller says something unrecognisable as a selection:\n"
            "    → Say ONLY: 'لطفاً یک عدد از یک تا نه انتخاب کنید: "
            "یک آفرها، دو انترنت، سه مکالمه، چهار بسته مختلط، پنج خدمات، "
            "شش پکیج، هفت موجودی، هشت کمک، نه DRM.'\n\n"

            "STEP B — CONFIRM SELECTION:\n"
            "  When caller says a valid option (number or service name), confirm it:\n"
            "  Example: 'شما گزینه دو، بسته‌های انترنت را انتخاب کردید. آیا درست است؟'\n"
            "  Wait for confirmation (بلی/نی). If they say no → go back to Step A.\n\n"

            "STEP C — HANDLE THE SELECTED OPTION:\n"
            "  After confirmation, provide brief helpful information about their selected service.\n"
            "  Answer follow-up questions ONLY within the scope of that selected option.\n\n"

            "OUT-OF-SCOPE RULE:\n"
            "  If the caller asks about something unrelated to Etisalat services, respond ONLY:\n"
            "  'بخشش می‌خواهم، من فقط می‌توانم در مورد خدمات اتصالات کمک کنم.'\n"
            "  Then re-prompt the menu once. Apply retry limit rules above.\n\n"

            "MENU OPTIONS (exact — never invent others):\n"
            "  ۱=بهترین آفرهای من  ۲=بسته‌های انترنت  ۳=بسته‌های مکالمه  "
            "۴=بسته‌های مختلط  ۵=خدمات  ۶=پکیج و مهاجرت  "
            "۷=پرسش موجودی  ۸=کمک بیشتر  ۹=غیرفعال کردن DRM\n\n"

            "LANGUAGE — MANDATORY: Respond ONLY in natural spoken Afghan Dari (دری افغانستانی). "
            "Never use Iranian Persian or English.\n\n"

            "AFGHAN DARI VOCABULARY — always use LEFT, never RIGHT:\n"
            "  استم/استی/استیم  →  NOT هستم/هستی/هستیم\n"
            "  بلی  →  NOT بله  |  بخشش می‌خواهم  →  NOT ببخشید\n"
            "  می‌خواهید  →  NOT در نظر دارید  |  وصل می‌کنم  →  NOT ارتباط می‌دهم\n"
            "  AVOID: چنین، مذکور، فوق، لذا، بنابراین\n\n"

            "CONVERSATION RULES:\n"
            "  - 1–2 short sentences max. Voice call — be brief.\n"
            "  - Use the FULL conversation history — never ask for info already given.\n"
            "  - Never use bullet points, lists, asterisks, or markdown.\n"
            "  - If ASR is clearly garbled (random noise), ask once to repeat. "
            "But if the message is intelligible, treat it as a selection attempt.\n"
            "  - Use the caller's name if they mention it."
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
        "mms_tts_model": "facebook/mms-tts-pps",   # Pashto (correct HuggingFace model ID)
        "mms_tts_sample_rate": 16_000,

        # TTS — edge-tts (fallback 1, Microsoft Azure)
        "edge_tts_voice": os.getenv("TTS_VOICE_PASHTO", "ps-AF-LatifaNeural"),
        "edge_tts_voice_male": "ps-AF-GulNawazNeural",

        # TTS — gTTS (fallback 2, Google — Pashto not supported; use Persian as closest)
        "gtts_language": "fa",

        # TTS — ElevenLabs (API-based, high quality Pashto)
        "elevenlabs_voice_id_male":   os.getenv("ELEVENLABS_VOICE_ID_PASHTO_MALE",   ""),
        "elevenlabs_voice_id_female": os.getenv("ELEVENLABS_VOICE_ID_PASHTO_FEMALE", ""),

        # TTS — Narakeet (REST API, Afghan Pashto)
        "narakeet_voice": os.getenv("NARAKEET_VOICE_PASHTO", "hamid"),   # ps-AF speaker

        # TTS — MicMonster (REST API)
        "micmonster_voice_id": os.getenv("MICMONSTER_VOICE_ID_PASHTO", ""),

        # TTS — Speakatoo (REST API)
        "speakatoo_voice_id": os.getenv("SPEAKATOO_VOICE_ID_PASHTO", ""),

        # LLM sentence boundaries (Arabic punctuation)
        "sentence_delimiters": (".", "!", "?", ",", "؟", "،"),

        # IVR Step 1 — Language selection
        "greeting": (
            "اتصالات ته ښه راغلاست. "
            "مهرباني وکړئ خپله ژبه وټاکئ. د ډیفالټ ژبه دري ده."
        ),

        # IVR Step 2 — Promotional announcement
        "ivr_promo": (
            "اتصالات پدې میاشت کې د انټرنیټ او غږ بنډلونو لپاره ځانګړي وړاندیزونه لري. "
            "د یو انتخاب کولو سره کولی شئ خپل غوره شخصي وړاندیزونه وګورئ."
        ),

        # IVR Steps 3–4 — Welcome greeting + main menu
        "ivr_main_menu": (
            "اتصالات ته ښه راغلاست! "
            "مهرباني وکړئ لاندې انتخابونو څخه یو غوره کړئ: "
            "یو د زما غوره وړاندیزونو لپاره، "
            "دوه د انټرنیټ بنډلونو لپاره، "
            "درې د غږ بنډلونو لپاره، "
            "څلور د مخلوط بنډلونو لپاره، "
            "پنځه د خدماتو لپاره، "
            "شپږ د پکیج او مهاجرت لپاره، "
            "اوه د بیلانس پوښتنې لپاره، "
            "اته د نورې مرستې لپاره، "
            "نهه د DRM بنډلونو د غیر فعالولو لپاره."
        ),

        # Stubs when Ollama is unreachable
        "neutral_stubs": [
            "بخښنه وغواړئ، یو شیبه صبر وکړئ.",
            "پوه شوم، اجازه راکړئ وګورم.",
            "مننه، یو شیبه صبر وکړئ.",
        ],

        # System persona — Etisalat Afghanistan Pashto IVR assistant (post-menu LLM handler)
        "system_prompt": (
            "You are Etisalat, the AI voice assistant for Etisalat Afghanistan telecom (IVR 888). "
            "You are on a live voice call. Your name is always Etisalat.\n\n"

            "CONTEXT: The IVR introduction has already been played — language selection, "
            "promotions, Etisalat greeting, and the 9-option main menu have all been presented. "
            "The caller must now choose one option from 1 to 9.\n\n"

            "EXIT INTENT — HIGHEST PRIORITY — CHECK FIRST EVERY TURN:\n"
            "  Before anything else, check if the caller wants to end the call.\n"
            "  Exit signals: خداحافظ، مننه بس دی، پای، لاړ شم، بیا به وګورم، "
            "نور کار نه لرم، ښه پاتې شئ، بس دی مننه\n"
            "  → Respond ONLY with: 'د اتصالات سره د اړیکې لپاره مننه. ښه ورځ ولرئ.'\n"
            "  → Do NOT re-prompt the menu. The conversation ends here.\n\n"

            "SMALL TALK / GREETINGS — CHECK SECOND:\n"
            "  If the caller says a greeting or social phrase (سلام، حال دې ښه دی، هلو، مننه، "
            "ستا غږ مې واورید) with no menu selection:\n"
            "  → Give ONE brief warm reply, then IMMEDIATELY re-prompt the menu.\n"
            "  Example: 'مننه! مهرباني وکړئ له یو تر نهه یو انتخاب وکړئ.'\n\n"

            "RETRY LIMIT — CHECK THIRD:\n"
            "  Read the FULL CONVERSATION HISTORY and count how many times you have already "
            "re-prompted the menu with no valid selection.\n"
            "  - After 2 re-prompts with no valid selection:\n"
            "    → 'که چیرې نورې مرستې ته اړتیا لرئ، کولی شم تاسو یو اپریټر سره وصل کړم. ایا غواړئ؟'\n"
            "  - After 3 re-prompts with no valid selection:\n"
            "    → 'د اتصالات سره د اړیکې لپاره مننه. ښه ورځ ولرئ.'\n"
            "    → End the conversation — do not loop further.\n\n"

            "STEP A — WAIT FOR SELECTION:\n"
            "  If none of the above apply, listen for a number 1–9 or a service name.\n"
            "  If the caller says something unrecognisable as a selection:\n"
            "    → Say ONLY: 'مهرباني وکړئ له یو څخه تر نهه یو شمیر ووایئ: "
            "یو وړاندیزونه، دوه انټرنیټ، درې غږ، څلور مخلوط، پنځه خدمات، "
            "شپږ پکیج، اوه بیلانس، اته مرسته، نهه DRM.'\n\n"

            "STEP B — CONFIRM SELECTION:\n"
            "  When caller says a valid option, confirm it:\n"
            "  Example: 'تاسو دوه، د انټرنیټ بنډلونه غوره کړل. ایا سمه ده؟'\n"
            "  Wait for confirmation (هو/نه). If they say no → go back to Step A.\n\n"

            "STEP C — HANDLE THE SELECTED OPTION:\n"
            "  After confirmation, provide brief helpful information about their selected service.\n"
            "  Answer follow-up questions ONLY within the scope of that selected option.\n\n"

            "OUT-OF-SCOPE RULE:\n"
            "  If the caller asks about something unrelated to Etisalat services, respond ONLY:\n"
            "  'بخښنه وغواړئ، زه یوازې د اتصالات خدماتو سره مرسته کولی شم.'\n"
            "  Then re-prompt the menu once. Apply retry limit rules above.\n\n"

            "MENU OPTIONS (exact — never invent others):\n"
            "  ۱=زما غوره وړاندیزونه  ۲=د انټرنیټ بنډلونه  ۳=د غږ بنډلونه  "
            "۴=مخلوط بنډلونه  ۵=خدمات  ۶=پکیج او مهاجرت  "
            "۷=د بیلانس پوښتنه  ۸=نوره مرسته  ۹=د DRM غیر فعالول\n\n"

            "LANGUAGE — MANDATORY: Always respond in Pashto (پښتو) script only. "
            "Never respond in English or any other language.\n\n"

            "CONVERSATION RULES:\n"
            "  - 1–2 short sentences max. Voice call — be brief.\n"
            "  - Use the FULL conversation history — never ask for info already given.\n"
            "  - Never use bullet points, lists, asterisks, or markdown.\n"
            "  - If ASR is clearly garbled (random noise), ask once to repeat. "
            "But if the message is intelligible, treat it as a selection attempt.\n"
            "  - Use the caller's name if they mention it."
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
    max_tokens: int = int(os.getenv("OLLAMA_MAX_TOKENS", "300"))


# ---------------------------------------------------------------------------
# RAG config
# ---------------------------------------------------------------------------

@dataclass
class RAGConfig:
    """FAISS-backed RAG for Etisalat Afghanistan IVR knowledge base."""
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
    max_turns: int = int(os.getenv("MEMORY_MAX_TURNS", "20"))


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
    # Pashto TTS engine priority for male voice
    # Comma-separated list tried in order until one succeeds.
    # Options: mms | elevenlabs | narakeet | micmonster | speakatoo | edge | gtts
    # Default: mms,edge,gtts  (same as before unless overridden)
    # ---------------------------------------------------------------------------
    pashto_engine_priority: str = os.getenv(
        "PASHTO_TTS_ENGINE_PRIORITY", "mms,edge,gtts"
    )

    # ---------------------------------------------------------------------------
    # Third-party TTS API keys
    # ---------------------------------------------------------------------------
    elevenlabs_api_key: str = field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))
    narakeet_api_key:   str = field(default_factory=lambda: os.getenv("NARAKEET_API_KEY",   ""))
    micmonster_api_key: str = field(default_factory=lambda: os.getenv("MICMONSTER_API_KEY", ""))
    speakatoo_api_key:  str = field(default_factory=lambda: os.getenv("SPEAKATOO_API_KEY",  ""))

    # ---------------------------------------------------------------------------
    # Debug audio dumps (optional, disabled by default)
    # ---------------------------------------------------------------------------
    debug_dump_audio: bool = os.getenv("DEBUG_TTS_DUMP_AUDIO", "false").strip().lower() in (
        "1", "true", "yes", "on"
    )
    debug_dump_dir: str = os.getenv("DEBUG_TTS_DUMP_DIR", "./debug_audio")


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
