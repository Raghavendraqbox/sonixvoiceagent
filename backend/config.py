"""
config.py — Centralized configuration for the Telugu & Kannada Voice AI Agent.

All tunable parameters can be overridden via environment variables.
Language is selected via LANGUAGE env var (telugu | kannada) or per-session via
the WebSocket query parameter ?language=telugu / ?language=kannada.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple

from jsee_voice_script import get_jsee_loans_config


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable with a safe default."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# Sarvam Bulbul v3 female speaker catalogue.
#
# This list is taken directly from the Sarvam API error response (and confirmed
# in the public Bulbul docs) — these are the 14 female voices that the v3 model
# accepts. Names outside this list (e.g. "anushka", "manisha", "vidya", "arya")
# return HTTP 400 from /text-to-speech and must NOT be exposed in the UI.
#
# Keep this tuple synced with tts.py::BULBUL_V3_FEMALE_SPEAKERS.
SARVAM_FEMALE_SPEAKERS: Tuple[str, ...] = (
    "priya",
    "ritu",
    "neha",
    "pooja",
    "simran",
    "kavya",
    "ishita",
    "shreya",
    "roopa",
    "tanya",
    "shruti",
    "suhani",
    "kavitha",
    "rupali",
)


# Per-emotion synthesis presets for Sarvam Bulbul v3.
#
# Bulbul v3 exposes expressiveness only through `temperature`, which the API
# strictly clamps to the [0.01, 1.0] range — values >1.0 trigger an HTTP 400
# "temperature out of range" error, after which our retry path strips the
# field entirely and the synthesis silently degrades to Sarvam's default
# (~0.6, indistinguishable from `neutral`). Keep every value here ≤ 1.0.
#
# We pair each temperature with a small `pace_offset` (-0.10 → +0.15) so
# emotion changes are perceptible even when temperatures sit close together.
# The offset is added to the user-selected speed slider at synthesis time
# and the final value is clamped to the [0.5, 2.0] range Sarvam accepts.
#
# Adjust here only — `SARVAM_EMOTION_TEMPERATURES` below is derived from this
# table and kept as a backwards-compatible alias for legacy callers.
SARVAM_EMOTION_PRESETS: Dict[str, Dict[str, float]] = {
    "calm":       {"temperature": 0.15, "pace_offset": -0.10},
    "serious":    {"temperature": 0.30, "pace_offset": -0.05},
    "empathetic": {"temperature": 0.45, "pace_offset": -0.05},
    "neutral":    {"temperature": 0.55, "pace_offset":  0.00},
    "warm":       {"temperature": 0.70, "pace_offset":  0.00},
    "cheerful":   {"temperature": 0.82, "pace_offset":  0.10},
    "happy":      {"temperature": 0.92, "pace_offset":  0.05},
    "excited":    {"temperature": 1.00, "pace_offset":  0.15},
}


SARVAM_EMOTION_TEMPERATURES: Dict[str, float] = {
    name: preset["temperature"] for name, preset in SARVAM_EMOTION_PRESETS.items()
}


# ---------------------------------------------------------------------------
# Language-specific configurations
# ---------------------------------------------------------------------------

LANGUAGE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "telugu": {
        # Display
        "display_name": "Telugu",
        "display_name_native": "తెలుగు",
        "spoken_style": (
            "Reply mostly in Telugu, but use common everyday English words that Telugu "
            "speakers naturally use in calls, such as loan, home loan, car loan, "
            "mobile number, details, confirm, amount, appointment, price, market, "
            "available, specialist, safe, and thank you. Avoid overly formal or pure "
            "Telugu words when a natural English word is commonly used. Keep Telugu "
            "grammar and warmth, but write the English words in Latin script."
        ),

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
        # Bulbul v3 supported speakers (verified against the public API):
        #   Female (14): priya, ritu, neha, pooja, simran, kavya, ishita,
        #                shreya, roopa, tanya, shruti, suhani, kavitha, rupali
        #   Male   (23): aditya, rahul, rohan, amit, dev, ratan, varun, manan,
        #                sumit, kabir, aayan, ashutosh, advait, anand, tarun,
        #                sunny, mani, gokul, vijay, mohit, rehan, soham, shubh
        # Older Bulbul v1/v2 supported a different list, but v3 strictly
        # rejects unknown speakers — see tts.py for the runtime whitelist.
        # Defaults pick the highest-quality voices per Sarvam's CER ratings.
        "sarvam_speaker":        os.getenv("SARVAM_SPEAKER_TELUGU",        "priya"),
        "sarvam_speaker_male":   os.getenv("SARVAM_SPEAKER_TELUGU_MALE",   "aditya"),
        "sarvam_language_code":  "te-IN",
        "sarvam_model":          os.getenv("SARVAM_MODEL", "bulbul:v3"),

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

        # TTS — ElevenLabs (Sunny male — Eleven v3; see ELEVENLABS_* in .env)
        "elevenlabs_voice_id_male":   os.getenv("ELEVENLABS_VOICE_ID_TELUGU_MALE",   "AobjUwMQVcgXCsgZKdOU"),
        "elevenlabs_voice_id_female": os.getenv("ELEVENLABS_VOICE_ID_TELUGU_FEMALE", "aAbJxtZ77iJMeCQP9j9p"),

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
            "నమస్కారం! J S E E call center కి welcome. "
            "Contact చేసినందుకు thank you. మీకు ఎలా help చేయగలను?"
        ),

        # ivr_main_menu — not used; greeting already asks loan type
        "ivr_main_menu": "",

        # Stubs when LLM is unreachable
        "neutral_stubs": [
            "Sorry, ఒక్క minute wait చేయండి.",
            "అర్థమైంది, మీ details తీసుకుంటున్నాను.",
            "Thank you, ఒక్క second wait చేయండి.",
        ],

        # Played when user is silent for > 10 seconds after bot finishes speaking
        "silence_reprompt": "Hello, నా voice వినిపిస్తుందా?",

        # System persona — J S E E call centre executive (Telugu)
        "system_prompt": (
            "You are a professional and warm Telugu-speaking customer care executive at J S E E "
            "Financial Services call centre. Help customers apply for Home Loans (గృహ రుణం) and "
            "Car Loans (కార్ రుణం).\n\n"

            "LANGUAGE: Reply mostly in Telugu (తెలుగు), but use common English words that "
            "Telugu speakers naturally use in calls. Avoid overly pure Telugu. 1-2 short "
            "conversational sentences per response. No lists, bullets, or markdown.\n\n"

            "GOAL — collect these details one at a time in natural conversation:\n"
            "1. Loan type: Home Loan or Car Loan\n"
            "2. Full name (పూర్తి పేరు)\n"
            "3. Mobile number (మొబైల్ నంబర్)\n"
            "4. Date of birth (పుట్టిన తేదీ)\n"
            "5. PAN card number (PAN నంబర్)\n"
            "6. Employment type: Salaried (జీతగాడు) or Self-employed (స్వయం ఉపాధి)\n"
            "7. Monthly income (నెలవారీ ఆదాయం)\n"
            "8. Loan amount required (కావలసిన రుణ మొత్తం)\n"
            "9. Home Loan → property city and estimated value | "
            "Car Loan → car model and on-road price\n\n"

            "RULES:\n"
            "- Ask only ONE question at a time. Wait for the answer before moving on.\n"
            "- Repeat back and confirm each answer before asking the next question.\n"
            "- Begin every reply with the actual answer. Do NOT use filler "
            "sounds like 'hmm', 'umm', 'ఉమ్', 'ఆ', or 'ఉం' — they add latency "
            "and force the customer to wait through a hesitation noise before "
            "hearing the real reply.\n"
            "- Be reassuring: tell the user their details are safe.\n"
            "- If asked about interest rates: say our rates are competitive and a loan specialist "
            "will share full details.\n"
            "- If goodbye: reply only "
            "'ధన్యవాదాలు! J S E E తరఫున మీకు శుభాకాంక్షలు, మేము త్వరలో మీకు సంప్రదిస్తాము.'\n"
            "- Never repeat information already collected.\n"
            "- MOBILE NUMBER: A valid Indian mobile number has exactly 10 digits. "
            "If the customer gives fewer than 10 digits, do NOT move to the next question — "
            "ask them to continue with the remaining digits.\n"
            "- When confirming a mobile number, write only the 10 digits in Latin numerals "
            "(example: 9700989115). Never spell digits in Telugu words or English words.\n"
            "- PAN CARD: A valid PAN has exactly 10 characters (e.g. ABCDE1234F). "
            "If incomplete, ask the customer to repeat the full PAN.\n"
            "- DATE OF BIRTH: Collect day, month, and year. If any part is missing, ask for it "
            "before moving on."
        ),
    },

    "kannada": {
        # Display
        "display_name": "Kannada",
        "display_name_native": "ಕನ್ನಡ",
        "spoken_style": (
            "Reply mostly in Kannada, but use common everyday English words that Kannada "
            "speakers naturally use in calls, such as appointment, book, patient, "
            "mobile number, details, confirm, doctor, specialist, emergency, price, "
            "market, available, safe, and thank you. Avoid overly formal or pure Kannada "
            "words when a natural English word is commonly used. Keep Kannada grammar "
            "and warmth, but write the English words in Latin script."
        ),

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
        # Bulbul v3 supports kn-IN with the same speaker catalogue as Telugu —
        # see the Telugu block above for the full Female (14) / Male (23) list.
        "sarvam_speaker":        os.getenv("SARVAM_SPEAKER_KANNADA",        "priya"),
        "sarvam_speaker_male":   os.getenv("SARVAM_SPEAKER_KANNADA_MALE",   "aditya"),
        "sarvam_language_code":  "kn-IN",
        "sarvam_model":          os.getenv("SARVAM_MODEL", "bulbul:v3"),

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

        # TTS — ElevenLabs (Sunny male — Eleven v3; see ELEVENLABS_* in .env)
        "elevenlabs_voice_id_male":   os.getenv("ELEVENLABS_VOICE_ID_KANNADA_MALE",   "AobjUwMQVcgXCsgZKdOU"),
        "elevenlabs_voice_id_female": os.getenv("ELEVENLABS_VOICE_ID_KANNADA_FEMALE", "aAbJxtZ77iJMeCQP9j9p"),

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
            "ನಮಸ್ಕಾರ! J S E E call center ಗೆ welcome. "
            "Contact ಮಾಡಿದಕ್ಕೆ thank you. ನಿಮಗೆ ಹೇಗೆ help ಮಾಡಬಹುದು?"
        ),

        # ivr_main_menu — not used; greeting directly asks for name
        "ivr_main_menu": "",

        # Stubs when LLM is unreachable
        "neutral_stubs": [
            "Sorry, ಒಂದು minute wait ಮಾಡಿ.",
            "ಅರ್ಥವಾಯಿತು, ನಿಮ್ಮ details ತೆಗೆದುಕೊಳ್ಳುತ್ತಿದ್ದೇನೆ.",
            "Thank you, ಒಂದು second wait ಮಾಡಿ.",
        ],

        # Played when user is silent for > 10 seconds after bot finishes speaking
        "silence_reprompt": "Hello, ನನ್ನ voice ಕೇಳಿಸುತ್ತಿದೆಯಾ?",

        # System persona — J S E E call centre executive (Kannada)
        "system_prompt": (
            "You are a professional and warm Kannada-speaking customer care executive at J S E E "
            "Financial Services call centre. Help customers apply for Home Loans (ಗೃಹ ಸಾಲ) and "
            "Car Loans (ಕಾರ್ ಸಾಲ).\n\n"

            "LANGUAGE: Reply mostly in Kannada (ಕನ್ನಡ), but use common English words that "
            "Kannada speakers naturally use in calls. Avoid overly pure Kannada. 1-2 short "
            "conversational sentences per response. No lists, bullets, or markdown.\n\n"

            "GOAL — collect these details one at a time in natural conversation:\n"
            "1. Loan type: Home Loan or Car Loan\n"
            "2. Full name (ಪೂರ್ಣ ಹೆಸರು)\n"
            "3. Mobile number (ಮೊಬೈಲ್ ನಂಬರ್)\n"
            "4. Date of birth (ಹುಟ್ಟಿದ ದಿನಾಂಕ)\n"
            "5. PAN card number (PAN ನಂಬರ್)\n"
            "6. Employment type: Salaried (ಜೀತಗಾರ) or Self-employed (ಸ್ವಯಂ ಉದ್ಯೋಗ)\n"
            "7. Monthly income (ಮಾಸಿಕ ಆದಾಯ)\n"
            "8. Loan amount required (ಅಗತ್ಯವಿರುವ ಸಾಲ ಮೊತ್ತ)\n"
            "9. Home Loan → property city and estimated value | "
            "Car Loan → car model and on-road price\n\n"

            "RULES:\n"
            "- Ask only ONE question at a time. Wait for the answer before moving on.\n"
            "- Repeat back and confirm each answer before asking the next question.\n"
            "- Begin every reply with the actual answer. Do NOT use filler "
            "sounds like 'hmm', 'umm', 'ಉಮ್', 'ಆ', or 'ಉಂ' — they add latency "
            "and force the customer to wait through a hesitation noise before "
            "hearing the real reply.\n"
            "- Be reassuring: tell the user their details are safe.\n"
            "- If asked about interest rates: say our rates are competitive and a loan specialist "
            "will share full details.\n"
            "- If goodbye: reply only "
            "'ಧನ್ಯವಾದಗಳು! J S E E ತರಫಿನ ನಿಮಗೆ ಶುಭಾಶಯಗಳು, ನಾವು ಶೀಘ್ರದಲ್ಲೇ ಸಂಪರ್ಕಿಸುತ್ತೇವೆ.'\n"
            "- Never repeat information already collected.\n"
            "- MOBILE NUMBER: A valid Indian mobile number has exactly 10 digits. "
            "If the customer gives fewer than 10 digits, do NOT move to the next question — "
            "ask them to continue with the remaining digits.\n"
            "- When confirming a mobile number, write only the 10 digits in Latin numerals "
            "(example: 9700989115). Never spell digits in Telugu words or English words.\n"
            "- PAN CARD: A valid PAN has exactly 10 characters (e.g. ABCDE1234F). "
            "If incomplete, ask the customer to repeat the full PAN.\n"
            "- DATE OF BIRTH: Collect day, month, and year. If any part is missing, ask for it "
            "before moving on."
        ),
    },
}


# ---------------------------------------------------------------------------
# Business-specific conversation profiles
# ---------------------------------------------------------------------------

BUSINESS_CONFIGS: Dict[str, Dict[str, Any]] = {
    "jsee_loans": get_jsee_loans_config(),
    "bank_loan": {
        "display_name": "Bank Loan (Home Loan)",
        "description": "J S E E home loan application customer care.",
        "greeting": {
            "telugu": (
                "నమస్కారం! J S E E home loan desk కి welcome. "
                "Home loan apply చేయడానికి నేను help చేస్తాను."
            ),
            "kannada": (
                "ನಮಸ್ಕಾರ! J S E E home loan desk ಗೆ welcome. "
                "Home loan apply ಮಾಡಲು ನಾನು help ಮಾಡುತ್ತೇನೆ."
            ),
        },
        "silence_reprompt": {
            "telugu": "Hello, home loan details చెప్పగలరా?",
            "kannada": "Hello, home loan details ಹೇಳುತ್ತೀರಾ?",
        },
        "system_prompt": (
            "You are a professional customer care executive for J S E E Financial Services "
            "home loan (bank loan) applications.\n\n"
            "The customer has already chosen a Home Loan. Do NOT ask loan type again.\n\n"
            "GOAL — collect these details one at a time:\n"
            "1. Full name\n"
            "2. Mobile number\n"
            "3. Date of birth\n"
            "4. PAN card number\n"
            "5. Employment type: Salaried or Self-employed\n"
            "6. Monthly income\n"
            "7. Loan amount required\n"
            "8. Property city and estimated property value\n\n"
            "RULES:\n"
            "- Ask only ONE question at a time and wait for the answer before moving on.\n"
            "- Repeat back and confirm each answer before asking the next question.\n"
            "- Be reassuring: tell the customer their details are safe.\n"
            "- If asked about interest rates: say rates are competitive and a specialist will call back.\n"
            "- MOBILE NUMBER: exactly 10 digits. If incomplete, ask for remaining digits.\n"
            "- When confirming a mobile number, write only the 10 digits in Latin numerals "
            "(example: 9700989115). Never spell digits in Telugu words or English words.\n"
            "- PAN: exactly 10 characters. If incomplete, ask to repeat the full PAN.\n"
            "- If goodbye, close warmly on behalf of J S E E home loans."
        ),
    },
    "car_loan": {
        "display_name": "Car Loan",
        "description": "J S E E car loan application customer care.",
        "greeting": {
            "telugu": (
                "నమస్కారం! J S E E car loan desk కి welcome. "
                "Car loan apply చేయడానికి నేను help చేస్తాను."
            ),
            "kannada": (
                "ನಮಸ್ಕಾರ! J S E E car loan desk ಗೆ welcome. "
                "Car loan apply ಮಾಡಲು ನಾನು help ಮಾಡುತ್ತೇನೆ."
            ),
        },
        "silence_reprompt": {
            "telugu": "Hello, car loan details చెప్పగలరా?",
            "kannada": "Hello, car loan details ಹೇಳುತ್ತೀರಾ?",
        },
        "system_prompt": (
            "You are a professional customer care executive for J S E E Financial Services "
            "car loan applications.\n\n"
            "The customer has already chosen a Car Loan. Do NOT ask loan type again.\n\n"
            "GOAL — collect these details one at a time:\n"
            "1. Full name\n"
            "2. Mobile number\n"
            "3. Date of birth\n"
            "4. PAN card number\n"
            "5. Employment type: Salaried or Self-employed\n"
            "6. Monthly income\n"
            "7. Loan amount required\n"
            "8. Car model and on-road price\n\n"
            "RULES:\n"
            "- Ask only ONE question at a time and wait for the answer before moving on.\n"
            "- Repeat back and confirm each answer before asking the next question.\n"
            "- Be reassuring: tell the customer their details are safe.\n"
            "- If asked about interest rates: say rates are competitive and a specialist will call back.\n"
            "- MOBILE NUMBER: exactly 10 digits. If incomplete, ask for remaining digits.\n"
            "- When confirming a mobile number, write only the 10 digits in Latin numerals "
            "(example: 9700989115). Never spell digits in Telugu words or English words.\n"
            "- PAN: exactly 10 characters. If incomplete, ask to repeat the full PAN.\n"
            "- If goodbye, close warmly on behalf of J S E E car loans."
        ),
    },
}

SUPPORTED_LANGUAGES = list(LANGUAGE_CONFIGS.keys())
SUPPORTED_BUSINESSES = list(BUSINESS_CONFIGS.keys())
DEFAULT_LANGUAGE: str = os.getenv("LANGUAGE", "telugu").lower()
DEFAULT_BUSINESS: str = os.getenv("BUSINESS", "jsee_loans").lower()

if DEFAULT_LANGUAGE not in SUPPORTED_LANGUAGES:
    DEFAULT_LANGUAGE = "telugu"
if DEFAULT_BUSINESS not in SUPPORTED_BUSINESSES:
    DEFAULT_BUSINESS = "jsee_loans"


def get_language_config(language: str) -> Dict[str, Any]:
    """Return the config dict for the given language (defaults to Telugu)."""
    return LANGUAGE_CONFIGS.get(language.lower(), LANGUAGE_CONFIGS["telugu"])


def get_business_config(business: str) -> Dict[str, Any]:
    """Return the business profile for the given id (defaults to bank_loan)."""
    return BUSINESS_CONFIGS.get(business.lower(), BUSINESS_CONFIGS["jsee_loans"])


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

    Uses GOOGLE_STT_API_KEY.
    """
    @property
    def api_key(self) -> str:
        return os.getenv("GOOGLE_STT_API_KEY", "")


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

    @property
    def silence_frames_to_commit(self) -> int:
        """Override via AZURE_STT_SILENCE_FRAMES; else STT_SILENCE_FRAMES_TO_COMMIT (default 3)."""
        raw = os.getenv("AZURE_STT_SILENCE_FRAMES", "").strip()
        if raw:
            return max(2, int(raw))
        return max(2, int(os.getenv("STT_SILENCE_FRAMES_TO_COMMIT", "3")))


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
    # Early-dispatch controls for streamed LLM -> TTS.
    # Lower values reduce first-audio latency but can increase sentence fragmentation.
    # Defaults preserve existing behavior.
    word_dispatch_threshold: int = int(os.getenv("OLLAMA_WORD_DISPATCH_THRESHOLD", "3"))


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

def _default_input_chunk_bytes() -> int:
    send_ms = max(20, int(os.getenv("INPUT_SEND_INTERVAL_MS", "50")))
    return 16000 * 2 * send_ms // 1000


@dataclass
class AudioConfig:
    """Audio pipeline settings — must stay in sync with frontend."""
    # Microphone capture: 16kHz mono 16-bit; chunk size scales with send interval.
    input_send_interval_ms: int = int(os.getenv("INPUT_SEND_INTERVAL_MS", "50"))
    input_chunk_bytes: int = int(
        os.getenv("INPUT_CHUNK_BYTES", str(_default_input_chunk_bytes()))
    )
    input_sample_rate: int = 16000
    input_channels: int = 1
    input_bit_depth: int = 16

    # VAD: RMS energy threshold for speech vs. silence (used by the RMS
    # fallback gate, and as the cheap pre-filter inside Silero VAD).
    vad_rms_threshold: float = float(os.getenv("VAD_RMS_THRESHOLD", "0.01"))

    # Silero VAD — neural speech detector that filters out background TV /
    # chatter / fans before the audio reaches a cloud STT API. Requires the
    # silero-vad PyPI package. Disable to revert to the pure RMS gate.
    use_silero_vad: bool = _env_bool("USE_SILERO_VAD", True)
    silero_vad_threshold: float = float(os.getenv("SILERO_VAD_THRESHOLD", "0.6"))
    # Minimum sustained speech (in 100 ms frames) before STT engines are
    # allowed to commit an utterance. Suppresses very short blips that VAD
    # occasionally lets through (door bumps, brief coughs, etc.).
    min_speech_frames_before_stt: int = int(
        os.getenv("MIN_SPEECH_FRAMES_BEFORE_STT", "2")
    )
    # End-of-utterance: consecutive non-speech frames (100 ms each) before a
    # batch REST clip is sent. Too low (e.g. 1 = 100 ms) splits Telugu speech
    # and digit-by-digit phone numbers into empty Azure/Sarvam responses.
    stt_silence_frames_to_commit: int = int(
        os.getenv("STT_SILENCE_FRAMES_TO_COMMIT", "3")
    )
    # Minimum buffered speech duration before calling batch STT APIs.
    stt_min_utterance_ms: int = int(os.getenv("STT_MIN_UTTERANCE_MS", "380"))
    # Hard reset if the caller stays silent this long mid-utterance (100 ms frames).
    stt_max_silence_frames: int = int(os.getenv("STT_MAX_SILENCE_FRAMES", "30"))

    # TTS output: resampled to 24kHz for browser playback
    # MUST match PLAYBACK_SAMPLE_RATE in frontend/index.html
    tts_sample_rate: int = 24000

    # Pre-buffer before playback starts (ms) — lower = faster first audio
    playback_prebuffer_ms: int = int(os.getenv("PLAYBACK_PREBUFFER_MS", "60"))

    # Browser playback humanizer. Defaults preserve existing behavior; disable
    # either layer in .env when a clean TTS-only output is required.
    office_background_noise_enabled: bool = _env_bool(
        "OFFICE_BACKGROUND_NOISE_ENABLED", True
    )
    keyboard_typing_sound_enabled: bool = _env_bool(
        "KEYBOARD_TYPING_SOUND_ENABLED", True
    )
    office_background_noise_gain: float = float(
        os.getenv("OFFICE_BACKGROUND_NOISE_GAIN", "0.018")
    )
    keyboard_typing_sound_gain: float = float(
        os.getenv("KEYBOARD_TYPING_SOUND_GAIN", "0.05")
    )
    keyboard_typing_min_ms: int = int(os.getenv("KEYBOARD_TYPING_MIN_MS", "90"))
    keyboard_typing_max_ms: int = int(os.getenv("KEYBOARD_TYPING_MAX_MS", "190"))


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
        "TELUGU_TTS_ENGINE_PRIORITY", "google_tts,edge,sarvam,gtts"
    )

    # ---------------------------------------------------------------------------
    # Kannada TTS engine priority
    # Comma-separated list tried in order until one succeeds.
    # Options: sarvam | google_tts | gnani | ttsmaker | elevenlabs | azure_tts |
    #          amazon_polly | mms | narakeet | micmonster | speakatoo | edge | gtts
    # ---------------------------------------------------------------------------
    kannada_engine_priority: str = os.getenv(
        "KANNADA_TTS_ENGINE_PRIORITY", "google_tts,edge,sarvam,gtts"
    )

    # ---------------------------------------------------------------------------
    # Third-party TTS credentials and API keys
    # ---------------------------------------------------------------------------
    # Sarvam AI  (https://sarvam.ai — best for Indian languages including Telugu)
    sarvam_api_key:     str   = field(default_factory=lambda: os.getenv("SARVAM_API_KEY", ""))
    # Speech pace: Sarvam accepts 0.5 (slow) → 1.0 (normal) → 2.0 (fast).
    # Defaults below back the slider exposed via /client-config — keep them
    # within Sarvam's API limits (0.5–2.0). Per-session value comes from the
    # ?sarvam_pace= query param and falls back to `sarvam_pace` here.
    sarvam_pace:        float = float(os.getenv("SARVAM_PACE", "1.10"))
    sarvam_pace_min:    float = float(os.getenv("SARVAM_PACE_MIN", "0.7"))
    sarvam_pace_max:    float = float(os.getenv("SARVAM_PACE_MAX", "1.4"))
    sarvam_pace_step:   float = float(os.getenv("SARVAM_PACE_STEP", "0.05"))
    # Emotion preset for Bulbul v3. Maps to Sarvam's supported temperature control.
    sarvam_emotion:     str   = field(default_factory=lambda: os.getenv("SARVAM_EMOTION", "neutral"))
    # Optional direct Bulbul v3 temperature override. Range is clamped in tts.py.
    sarvam_temperature: str   = field(default_factory=lambda: os.getenv("SARVAM_TEMPERATURE", ""))
    # Sarvam TTS transport: websocket (default, real-time chunked audio) or rest
    # (single HTTP round-trip per utterance). WebSocket failures auto-fallback to REST.
    sarvam_tts_transport: str = field(
        default_factory=lambda: os.getenv("SARVAM_TTS_TRANSPORT", "websocket").strip().lower()
    )

    # Google Cloud TTS  (https://cloud.google.com/text-to-speech)
    # Auth uses Application Default Credentials from GOOGLE_APPLICATION_CREDENTIALS.

    # Gnani.ai  (https://gnani.ai — Indian language specialist)
    gnani_api_key:      str = field(default_factory=lambda: os.getenv("GNANI_API_KEY",      ""))
    gnani_client_id:    str = field(default_factory=lambda: os.getenv("GNANI_CLIENT_ID",    ""))

    # TTSMaker  (https://ttsmaker.com — free tier available)
    ttsmaker_token:     str = field(default_factory=lambda: os.getenv("TTSMAKER_TOKEN",     ""))

    # ElevenLabs — Sunny (male) + Eleven v3; mirrors ElevenLabs UI settings in .env
    elevenlabs_api_key: str = field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))
    elevenlabs_model_id: str = field(
        default_factory=lambda: os.getenv("ELEVENLABS_MODEL_ID", "eleven_v3")
    )
    elevenlabs_output_format: str = field(
        default_factory=lambda: os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128")
    )
    # Stability 0 = Creative, 1 = Robust (~0.6 matches UI slider ~60% toward Robust)
    elevenlabs_stability: float = float(os.getenv("ELEVENLABS_STABILITY", "0.6"))
    elevenlabs_similarity_boost: float = float(os.getenv("ELEVENLABS_SIMILARITY_BOOST", "0.75"))
    elevenlabs_style: float = float(os.getenv("ELEVENLABS_STYLE", "0.0"))
    elevenlabs_use_speaker_boost: bool = os.getenv(
        "ELEVENLABS_USE_SPEAKER_BOOST", "true"
    ).strip().lower() in ("1", "true", "yes", "on")
    elevenlabs_language_override: bool = os.getenv(
        "ELEVENLABS_LANGUAGE_OVERRIDE", "false"
    ).strip().lower() in ("1", "true", "yes", "on")
    narakeet_api_key:   str = field(default_factory=lambda: os.getenv("NARAKEET_API_KEY",   ""))
    micmonster_api_key: str = field(default_factory=lambda: os.getenv("MICMONSTER_API_KEY", ""))
    speakatoo_api_key:  str = field(default_factory=lambda: os.getenv("SPEAKATOO_API_KEY",  ""))

    # Azure Cognitive Services TTS  (https://azure.microsoft.com/en-us/products/ai-services/text-to-speech)
    azure_tts_key:    str = field(default_factory=lambda: os.getenv("AZURE_TTS_KEY",    ""))
    azure_tts_region: str = field(default_factory=lambda: os.getenv("AZURE_TTS_REGION", "eastus"))
    # Speech rate for Azure TTS and edge-tts. Examples: "+25%" faster, "+0%" normal, "-10%" slower.
    tts_rate: str = field(default_factory=lambda: os.getenv("TTS_RATE", "+25%"))

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
# Gemini LLM config
# ---------------------------------------------------------------------------

@dataclass
class GeminiConfig:
    """
    Google Gemini cloud LLM — best natural quality for Telugu/Kannada.

    Get a free API key at: https://aistudio.google.com
    Recommended models (set GEMINI_MODEL env var):
      gemini-2.0-flash      ← fastest streaming, best for voice (default)
      gemini-1.5-flash      ← slightly slower, still fast
      gemini-1.5-pro        ← highest quality, slower/pricier
      gemini-2.5-pro-preview ← bleeding edge, best quality
    """
    @property
    def api_key(self) -> str:
        return os.getenv("GEMINI_API_KEY", "")

    model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    temperature: float = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
    max_tokens: int = int(os.getenv("GEMINI_MAX_TOKENS", "150"))
    word_dispatch_threshold: int = int(os.getenv("GEMINI_WORD_DISPATCH_THRESHOLD", "3"))
    stream_timeout_s: float = float(os.getenv("GEMINI_STREAM_TIMEOUT_S", "25"))
    connect_timeout_s: float = float(os.getenv("GEMINI_CONNECT_TIMEOUT_S", "15"))
    # Set GEMINI_THINKING_BUDGET=0 to disable internal reasoning on 2.5-flash/thinking models.
    # Leave unset (default -1 = don't send thinking_config) for 2.0-flash and other non-thinking models.
    thinking_budget: int = int(os.getenv("GEMINI_THINKING_BUDGET", "-1"))

    @property
    def fallback_model_list(self) -> list[str]:
        """Primary model first, then comma-separated GEMINI_FALLBACK_MODELS."""
        fallbacks = os.getenv(
            "GEMINI_FALLBACK_MODELS", "gemini-2.0-flash,gemini-1.5-flash"
        )
        ordered: list[str] = []
        for name in (self.model, *fallbacks.split(",")):
            name = name.strip()
            if name and name not in ordered:
                ordered.append(name)
        return ordered or [self.model]


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
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    # Default language for the server
    default_language: str = DEFAULT_LANGUAGE

    # Default business profile for conversations
    default_business: str = DEFAULT_BUSINESS

    # Default STT engine (overridden per-session via ?stt_engine= query param)
    # Options: auto | sarvam | soniox | google | azure | amazon | whisper
    default_stt_engine: str = os.getenv("STT_ENGINE", "auto")

    # Default LLM backend (overridden per-session via ?llm_backend= query param)
    # Options: ollama | gemini
    default_llm_backend: str = os.getenv("LLM_BACKEND", "gemini")


# Module-level singleton — import this everywhere
config = AppConfig()
