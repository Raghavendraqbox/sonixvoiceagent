"""
config.py — Centralized configuration for the Telugu & Kannada Voice AI Agent.

All tunable parameters can be overridden via environment variables.
Language is selected via LANGUAGE env var (telugu | kannada) or per-session via
the WebSocket query parameter ?language=telugu / ?language=kannada.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple


# ---------------------------------------------------------------------------
# Language-specific configurations
# ---------------------------------------------------------------------------

LANGUAGE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "telugu": {
        # Display
        "display_name": "Telugu",
        "display_name_native": "తెలుగు",

        # ASR
        "soniox_language_code": "te",    # Telugu language code
        "whisper_language": "te",         # Whisper language code for Telugu
        "sarvam_stt_language_code":         "te-IN",  # Sarvam STT language code
        "google_stt_language_code":         "te-IN",  # Google Cloud STT language code
        "azure_stt_language_code":          "te-IN",  # Azure STT language code
        "amazon_transcribe_language_code":  "te-IN",  # Amazon Transcribe language code

        # TTS — Meta MMS-TTS (local GPU)
        # facebook/mms-tts-tel — ISO 639-3: tel = Telugu
        "mms_tts_model": "facebook/mms-tts-tel",
        "mms_tts_sample_rate": 16_000,

        # TTS — Sarvam AI (https://sarvam.ai)
        # Female speakers: anushka, manisha, vidya, arya, ritu, priya, neha, pooja, simra, kavya,
        #                  ishita, shreya, roopa, tanya, sunny, suhani, kavitha, rupal
        # Male speakers:   abhilash, karun, hitesh, aditya, rahul, rohan, amit, dev, ratan, varun,
        #                  manan, sumit, kabir, aayan, shubh, ashutosh, advait, anand, tarun, mani,
        #                  gokul, vijay, mohit, rehan, soham
        "sarvam_speaker":        os.getenv("SARVAM_SPEAKER_TELUGU",        "anushka"),
        "sarvam_speaker_male":   os.getenv("SARVAM_SPEAKER_TELUGU_MALE",   "abhilash"),
        "sarvam_language_code":  "te-IN",
        "sarvam_model":          os.getenv("SARVAM_MODEL", "bulbul:v2"),

        # TTS — Google Cloud TTS (https://cloud.google.com/text-to-speech)
        # Telugu voices: te-IN-Standard-A (female), te-IN-Standard-B (male)
        #                te-IN-Wavenet-A  (female), te-IN-Wavenet-B  (male)
        "google_tts_voice":        os.getenv("GOOGLE_TTS_VOICE_TELUGU",      "te-IN-Standard-A"),
        "google_tts_voice_male":   os.getenv("GOOGLE_TTS_VOICE_TELUGU_MALE", "te-IN-Standard-B"),
        "google_tts_language":     "te-IN",

        # TTS — Gnani.ai (https://gnani.ai)
        "gnani_language_code":   "te",
        "gnani_voice":           os.getenv("GNANI_VOICE_TELUGU", "female"),

        # TTS — TTSMaker (https://ttsmaker.com)
        # Find Telugu voice IDs at: https://api.ttsmaker.com/v1/get-voice-list
        "ttsmaker_voice_id":     int(os.getenv("TTSMAKER_VOICE_ID_TELUGU", "0")),

        # TTS — ElevenLabs (legacy fallback)
        # iP95p4xoKVk53GoZ742B = Chris (male, multilingual v2 compatible)
        # onwK4e9ZLuTAKqWW03F9 = River (female, multilingual v2, best quality)
        "elevenlabs_voice_id_male":   os.getenv("ELEVENLABS_VOICE_ID_TELUGU_MALE",   "iP95p4xoKVk53GoZ742B"),
        "elevenlabs_voice_id_female": os.getenv("ELEVENLABS_VOICE_ID_TELUGU_FEMALE", "onwK4e9ZLuTAKqWW03F9"),

        # TTS — Azure Cognitive Services (https://azure.microsoft.com/en-us/products/ai-services/text-to-speech)
        # Telugu neural voices: te-IN-ShrutiNeural (female), te-IN-MohanNeural (male)
        "azure_tts_voice":      os.getenv("AZURE_TTS_VOICE_TELUGU",      "te-IN-ShrutiNeural"),
        "azure_tts_voice_male": os.getenv("AZURE_TTS_VOICE_TELUGU_MALE", "te-IN-MohanNeural"),
        "azure_tts_language":   "te-IN",

        # TTS — Amazon Polly (https://aws.amazon.com/polly/)
        # Telugu voice: set AWS_POLLY_VOICE_TELUGU to a valid Polly voice ID.
        # Polly does not have a dedicated Telugu voice; use "Aditi" (hi-IN, Indian accent)
        # or check https://docs.aws.amazon.com/polly/latest/dg/voicelist.html for updates.
        "amazon_polly_voice":          os.getenv("AWS_POLLY_VOICE_TELUGU",        "Aditi"),
        "amazon_polly_voice_male":     os.getenv("AWS_POLLY_VOICE_TELUGU_MALE",   "Aditi"),
        "amazon_polly_language_code":  os.getenv("AWS_POLLY_LANGUAGE_TELUGU",     "hi-IN"),
        "amazon_polly_engine":         os.getenv("AWS_POLLY_ENGINE",              "standard"),

        # TTS — edge-tts (free Microsoft neural voices)
        "edge_tts_voice": os.getenv("TTS_VOICE_TELUGU", "te-IN-ShrutiNeural"),
        "edge_tts_voice_male": "te-IN-MohanNeural",

        # TTS — gTTS (last resort fallback)
        "gtts_language": "te",

        # LLM sentence boundaries
        "sentence_delimiters": (".", "!", "?", ",", "।", "?"),

        # Greeting — plays once on the first user utterance
        "greeting": (
            "నమస్కారం! నేను మీ AI వాయిస్ అసిస్టెంట్‌ను. "
            "నేను తెలుగులో మీకు సహాయం చేయగలను. మీకు ఏమి అవసరం?"
        ),

        # ivr_main_menu — leave empty; no IVR menu in generic mode
        "ivr_main_menu": "",

        # Stubs when Ollama is unreachable
        "neutral_stubs": [
            "క్షమించండి, ఒక్క నిమిషం ఆగండి.",
            "అర్థమైంది, చూడనివ్వండి.",
            "చాలా ధన్యవాదాలు, ఒక్క నిమిషం ఆగండి.",
        ],

        # System persona — generic helpful Telugu voice assistant
        "system_prompt": (
            "Telugu voice assistant on a live call. Reply ONLY in Telugu (తెలుగు). "
            "1-2 short sentences max. No lists, bullets, or markdown. "
            "If goodbye: reply only 'ధన్యవాదాలు! మీ రోజు శుభంగా గడవాలి.' "
            "Never repeat info already given. Be warm and direct."
        ),
    },

    "kannada": {
        # Display
        "display_name": "Kannada",
        "display_name_native": "ಕನ್ನಡ",

        # ASR
        "soniox_language_code": "kn",    # Kannada language code
        "whisper_language": "kn",         # faster-whisper supports Kannada
        "sarvam_stt_language_code":         "kn-IN",  # Sarvam STT language code
        "google_stt_language_code":         "kn-IN",  # Google Cloud STT language code
        "azure_stt_language_code":          "kn-IN",  # Azure STT language code
        "amazon_transcribe_language_code":  "kn-IN",  # Amazon Transcribe language code

        # TTS — Meta MMS-TTS (local GPU)
        "mms_tts_model": "facebook/mms-tts-kan",   # Kannada (HuggingFace model ID)
        "mms_tts_sample_rate": 16_000,

        # TTS — Sarvam AI (https://sarvam.ai — best for Indian languages including Kannada)
        # Same speaker pool as Telugu; bulbul:v2 supports kn-IN
        "sarvam_speaker":        os.getenv("SARVAM_SPEAKER_KANNADA",        "anushka"),
        "sarvam_speaker_male":   os.getenv("SARVAM_SPEAKER_KANNADA_MALE",   "abhilash"),
        "sarvam_language_code":  "kn-IN",
        "sarvam_model":          os.getenv("SARVAM_MODEL", "bulbul:v2"),

        # TTS — Google Cloud TTS (https://cloud.google.com/text-to-speech)
        # Kannada voices: kn-IN-Standard-A (female), kn-IN-Standard-B (male)
        #                 kn-IN-Wavenet-A  (female), kn-IN-Wavenet-B  (male)
        "google_tts_voice":        os.getenv("GOOGLE_TTS_VOICE_KANNADA",      "kn-IN-Standard-A"),
        "google_tts_voice_male":   os.getenv("GOOGLE_TTS_VOICE_KANNADA_MALE", "kn-IN-Standard-B"),
        "google_tts_language":     "kn-IN",

        # TTS — Gnani.ai (https://gnani.ai — Indian language specialist)
        "gnani_language_code":   "kn",
        "gnani_voice":           os.getenv("GNANI_VOICE_KANNADA", "female"),

        # TTS — TTSMaker (https://ttsmaker.com — free tier available)
        # Find Kannada voice IDs at: https://api.ttsmaker.com/v1/get-voice-list
        "ttsmaker_voice_id":     int(os.getenv("TTSMAKER_VOICE_ID_KANNADA", "0")),

        # TTS — ElevenLabs (API-based, high quality Kannada)
        "elevenlabs_voice_id_male":   os.getenv("ELEVENLABS_VOICE_ID_KANNADA_MALE",   "iP95p4xoKVk53GoZ742B"),
        "elevenlabs_voice_id_female": os.getenv("ELEVENLABS_VOICE_ID_KANNADA_FEMALE", "onwK4e9ZLuTAKqWW03F9"),

        # TTS — Azure Cognitive Services (https://azure.microsoft.com/en-us/products/ai-services/text-to-speech)
        # Kannada neural voices: kn-IN-SapnaNeural (female), kn-IN-GaganNeural (male)
        "azure_tts_voice":      os.getenv("AZURE_TTS_VOICE_KANNADA",      "kn-IN-SapnaNeural"),
        "azure_tts_voice_male": os.getenv("AZURE_TTS_VOICE_KANNADA_MALE", "kn-IN-GaganNeural"),
        "azure_tts_language":   "kn-IN",

        # TTS — Amazon Polly (https://aws.amazon.com/polly/)
        # Polly does not have a dedicated Kannada voice; use "Aditi" (hi-IN, Indian accent)
        # or check https://docs.aws.amazon.com/polly/latest/dg/voicelist.html for updates.
        "amazon_polly_voice":          os.getenv("AWS_POLLY_VOICE_KANNADA",        "Aditi"),
        "amazon_polly_voice_male":     os.getenv("AWS_POLLY_VOICE_KANNADA_MALE",   "Aditi"),
        "amazon_polly_language_code":  os.getenv("AWS_POLLY_LANGUAGE_KANNADA",     "hi-IN"),
        "amazon_polly_engine":         os.getenv("AWS_POLLY_ENGINE",               "standard"),

        # TTS — Narakeet (REST API)
        "narakeet_voice": os.getenv("NARAKEET_VOICE_KANNADA", ""),

        # TTS — MicMonster (REST API)
        "micmonster_voice_id": os.getenv("MICMONSTER_VOICE_ID_KANNADA", ""),

        # TTS — Speakatoo (REST API)
        "speakatoo_voice_id": os.getenv("SPEAKATOO_VOICE_ID_KANNADA", ""),

        # TTS — edge-tts (free Microsoft neural voices)
        "edge_tts_voice": os.getenv("TTS_VOICE_KANNADA", "kn-IN-SapnaNeural"),
        "edge_tts_voice_male": "kn-IN-GaganNeural",

        # TTS — gTTS (last resort fallback)
        "gtts_language": "kn",

        # LLM sentence boundaries
        "sentence_delimiters": (".", "!", "?", ",", "।", "?"),

        # Greeting — plays once on the first user utterance
        "greeting": (
            "ನಮಸ್ಕಾರ! ನಾನು ನಿಮ್ಮ AI ವಾಯ್ಸ್ ಅಸಿಸ್ಟೆಂಟ್. "
            "ನಾನು ಕನ್ನಡದಲ್ಲಿ ನಿಮಗೆ ಸಹಾಯ ಮಾಡಬಲ್ಲೆ. ನಿಮಗೆ ಏನು ಬೇಕು?"
        ),

        # ivr_main_menu — leave empty; no IVR menu in generic mode
        "ivr_main_menu": "",

        # Stubs when Ollama is unreachable
        "neutral_stubs": [
            "ಕ್ಷಮಿಸಿ, ಒಂದು ಕ್ಷಣ ಕಾಯಿರಿ.",
            "ಅರ್ಥವಾಯಿತು, ಪರಿಶೀಲಿಸುತ್ತೇನೆ.",
            "ತುಂಬಾ ಧನ್ಯವಾದಗಳು, ಒಂದು ಕ್ಷಣ ಕಾಯಿರಿ.",
        ],

        # System persona — generic helpful Kannada voice assistant
        "system_prompt": (
            "Kannada voice assistant on a live call. Reply ONLY in Kannada (ಕನ್ನಡ). "
            "1-2 short sentences max. No lists, bullets, or markdown. "
            "If goodbye: reply only 'ಧನ್ಯವಾದಗಳು! ನಿಮ್ಮ ದಿನ ಚೆನ್ನಾಗಿ ಕಳೆಯಲಿ.' "
            "Never repeat info already given. Be warm and direct."
        ),
    },
}

SUPPORTED_LANGUAGES = list(LANGUAGE_CONFIGS.keys())
DEFAULT_LANGUAGE: str = os.getenv("LANGUAGE", "telugu").lower()

if DEFAULT_LANGUAGE not in SUPPORTED_LANGUAGES:
    DEFAULT_LANGUAGE = "telugu"


def get_language_config(language: str) -> Dict[str, Any]:
    """Return the config dict for the given language (defaults to Telugu)."""
    return LANGUAGE_CONFIGS.get(language.lower(), LANGUAGE_CONFIGS["telugu"])


# ---------------------------------------------------------------------------
# Sarvam STT config
# ---------------------------------------------------------------------------

@dataclass
class SarvamSTTConfig:
    """
    Sarvam AI speech-to-text — cloud API, best for Indian languages.

    Endpoint: POST https://api.sarvam.ai/speech-to-text
    Auth:     api-subscription-key header (same key as Sarvam TTS)
    Model:    saarika:v2.5  (v1/v2 are deprecated)
    Accepts:  multipart/form-data — file (WAV), language_code, model
    """
    # Reuse the same SARVAM_API_KEY used by TTS
    @property
    def api_key(self) -> str:
        return os.getenv("SARVAM_API_KEY", "")

    model: str = os.getenv("SARVAM_STT_MODEL", "saarika:v2.5")
    endpoint: str = "https://api.sarvam.ai/speech-to-text"


# ---------------------------------------------------------------------------
# Soniox streaming ASR config
# ---------------------------------------------------------------------------

@dataclass
class SonioxConfig:
    """Soniox streaming ASR — cloud-based, supports Telugu (te) and Kannada (kn)."""

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
# Google Cloud Speech-to-Text config
# ---------------------------------------------------------------------------

@dataclass
class GoogleSTTConfig:
    """
    Google Cloud Speech-to-Text — REST API, supports te-IN and kn-IN.

    Endpoint: POST https://speech.googleapis.com/v1/speech:recognize?key={api_key}
    Accepts:  LINEAR16 WAV, 16 kHz mono
    Docs:     https://cloud.google.com/speech-to-text/docs/reference/rest/v1/speech/recognize

    Uses GOOGLE_STT_API_KEY (or GOOGLE_TTS_API_KEY as fallback — same GCP project key works).
    """
    @property
    def api_key(self) -> str:
        return os.getenv("GOOGLE_STT_API_KEY") or os.getenv("GOOGLE_TTS_API_KEY", "")


# ---------------------------------------------------------------------------
# Azure Cognitive Services Speech-to-Text config
# ---------------------------------------------------------------------------

@dataclass
class AzureSTTConfig:
    """
    Microsoft Azure Speech-to-Text — REST API, supports te-IN and kn-IN.

    Endpoint: POST https://{region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1
    Accepts:  WAV audio/wav, 16 kHz mono
    Docs:     https://learn.microsoft.com/en-us/azure/ai-services/speech-service/rest-speech-to-text

    Uses AZURE_STT_KEY (or AZURE_TTS_KEY fallback) and AZURE_STT_REGION (or AZURE_TTS_REGION fallback).
    """
    @property
    def api_key(self) -> str:
        return os.getenv("AZURE_STT_KEY") or os.getenv("AZURE_TTS_KEY", "")

    @property
    def region(self) -> str:
        return os.getenv("AZURE_STT_REGION") or os.getenv("AZURE_TTS_REGION", "eastus")


# ---------------------------------------------------------------------------
# Amazon Transcribe config
# ---------------------------------------------------------------------------

@dataclass
class AmazonTranscribeConfig:
    """
    Amazon Transcribe Streaming — supports te-IN and kn-IN.

    Uses amazon-transcribe Python package (pip install amazon-transcribe).
    Shares AWS credentials with Amazon Polly TTS.
    Docs: https://docs.aws.amazon.com/transcribe/latest/dg/streaming.html
    """
    @property
    def access_key(self) -> str:
        return os.getenv("AWS_ACCESS_KEY_ID", "")

    @property
    def secret_key(self) -> str:
        return os.getenv("AWS_SECRET_ACCESS_KEY", "")

    @property
    def region(self) -> str:
        return os.getenv("AWS_REGION_NAME", "us-east-1")


# ---------------------------------------------------------------------------
# Ollama LLM config
# ---------------------------------------------------------------------------

@dataclass
class OllamaConfig:
    """
    Ollama local LLM — open-source, no API key required.

    Recommended models for Telugu/Kannada (pull before starting):
      qwen2.5:7b    → ~6 GB VRAM   ← good multilingual quality
      qwen2.5:14b   → ~10 GB VRAM  ← better multilingual quality
      qwen2.5:32b   → ~20 GB VRAM  ← best quality
      qwen2.5:72b   → ~48 GB VRAM  ← best overall (default for 80GB GPU)
      aya-expanse   → varies       ← Cohere multilingual, Indian languages supported

    Pull the model before starting:
      ollama pull qwen2.5:7b
    """
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:72b")
    temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
    top_p: float = float(os.getenv("OLLAMA_TOP_P", "0.9"))
    max_tokens: int = int(os.getenv("OLLAMA_MAX_TOKENS", "200"))
    # Keep context small for voice — large context (Ollama default: 262144) causes
    # 8-10s prefill on 72b models. 4096 is more than enough for voice conversations.
    num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "4096"))


# ---------------------------------------------------------------------------
# RAG config
# ---------------------------------------------------------------------------

@dataclass
class RAGConfig:
    """FAISS-backed RAG for the voice assistant knowledge base."""
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
    # Telugu TTS engine priority
    # Comma-separated list tried in order until one succeeds.
    # Options: sarvam | google_tts | gnani | ttsmaker | elevenlabs | edge | gtts
    # Default: sarvam,google_tts,gnani,ttsmaker,edge,gtts
    # ---------------------------------------------------------------------------
    telugu_engine_priority: str = os.getenv(
        "TELUGU_TTS_ENGINE_PRIORITY", "edge,sarvam,gtts"
    )

    # ---------------------------------------------------------------------------
    # Kannada TTS engine priority
    # Comma-separated list tried in order until one succeeds.
    # Options: sarvam | google_tts | gnani | ttsmaker | elevenlabs | azure_tts |
    #          amazon_polly | mms | narakeet | micmonster | speakatoo | edge | gtts
    # ---------------------------------------------------------------------------
    kannada_engine_priority: str = os.getenv(
        "KANNADA_TTS_ENGINE_PRIORITY", "edge,sarvam,gtts"
    )

    # ---------------------------------------------------------------------------
    # Third-party TTS API keys
    # ---------------------------------------------------------------------------
    # Sarvam AI  (https://sarvam.ai — best for Indian languages including Telugu)
    sarvam_api_key:     str   = field(default_factory=lambda: os.getenv("SARVAM_API_KEY", ""))
    # Speech pace: 0.5 (slow) → 1.0 (normal) → 2.0 (fast). Default 1.0.
    sarvam_pace:        float = float(os.getenv("SARVAM_PACE", "1.0"))

    # Google Cloud TTS  (https://cloud.google.com/text-to-speech)
    google_tts_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_TTS_API_KEY", ""))

    # Gnani.ai  (https://gnani.ai — Indian language specialist)
    gnani_api_key:      str = field(default_factory=lambda: os.getenv("GNANI_API_KEY",      ""))
    gnani_client_id:    str = field(default_factory=lambda: os.getenv("GNANI_CLIENT_ID",    ""))

    # TTSMaker  (https://ttsmaker.com — free tier available)
    ttsmaker_token:     str = field(default_factory=lambda: os.getenv("TTSMAKER_TOKEN",     ""))

    # ElevenLabs  (legacy / Kannada)
    elevenlabs_api_key: str = field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))
    narakeet_api_key:   str = field(default_factory=lambda: os.getenv("NARAKEET_API_KEY",   ""))
    micmonster_api_key: str = field(default_factory=lambda: os.getenv("MICMONSTER_API_KEY", ""))
    speakatoo_api_key:  str = field(default_factory=lambda: os.getenv("SPEAKATOO_API_KEY",  ""))

    # Azure Cognitive Services TTS  (https://azure.microsoft.com/en-us/products/ai-services/text-to-speech)
    azure_tts_key:    str = field(default_factory=lambda: os.getenv("AZURE_TTS_KEY",    ""))
    azure_tts_region: str = field(default_factory=lambda: os.getenv("AZURE_TTS_REGION", "eastus"))

    # Amazon Polly  (https://aws.amazon.com/polly/)
    amazon_polly_access_key: str = field(default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID",     ""))
    amazon_polly_secret_key: str = field(default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY", ""))
    amazon_polly_region:     str = field(default_factory=lambda: os.getenv("AWS_REGION_NAME",       "us-east-1"))

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
    sarvam_stt: SarvamSTTConfig = field(default_factory=SarvamSTTConfig)
    soniox: SonioxConfig = field(default_factory=SonioxConfig)
    google_stt: GoogleSTTConfig = field(default_factory=GoogleSTTConfig)
    azure_stt: AzureSTTConfig = field(default_factory=AzureSTTConfig)
    amazon_transcribe: AmazonTranscribeConfig = field(default_factory=AmazonTranscribeConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    # Default language for the server
    default_language: str = DEFAULT_LANGUAGE

    # Default STT engine (overridden per-session via ?stt_engine= query param)
    # Options: auto | sarvam | soniox | google | azure | amazon | whisper
    default_stt_engine: str = os.getenv("STT_ENGINE", "auto")


# Module-level singleton — import this everywhere
config = AppConfig()
