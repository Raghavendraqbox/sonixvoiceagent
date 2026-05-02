"""
config.py — Centralized configuration for the Telugu & Kannada Voice AI Agent.

All tunable parameters can be overridden via environment variables.
Language is selected via LANGUAGE env var (telugu | kannada) or per-session via
the WebSocket query parameter ?language=telugu / ?language=kannada.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple


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
# Bulbul v3 exposes expressiveness only through `temperature`, so to make the
# eight presets audibly distinct we widen the temperature spread (0.20 → 1.25)
# and couple a small `pace_offset` (-0.10 → +0.15) into each emotion. The
# offset is added to the user-selected speed slider value at synthesis time,
# clamped to the [0.5, 2.0] range Sarvam accepts.
#
# Adjust here only — `SARVAM_EMOTION_TEMPERATURES` below is derived from this
# table and kept as a backwards-compatible alias for legacy callers.
SARVAM_EMOTION_PRESETS: Dict[str, Dict[str, float]] = {
    "neutral":    {"temperature": 0.55, "pace_offset":  0.00},
    "calm":       {"temperature": 0.20, "pace_offset": -0.10},
    "warm":       {"temperature": 0.75, "pace_offset":  0.00},
    "empathetic": {"temperature": 0.45, "pace_offset": -0.05},
    "happy":      {"temperature": 0.95, "pace_offset":  0.05},
    "cheerful":   {"temperature": 0.85, "pace_offset":  0.10},
    "excited":    {"temperature": 1.25, "pace_offset":  0.15},
    "serious":    {"temperature": 0.30, "pace_offset": -0.05},
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
            "నమస్కారం! QOBOX call center కి welcome. "
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

        # System persona — QOBOX call centre executive (Telugu)
        "system_prompt": (
            "You are a professional and warm Telugu-speaking customer care executive at QOBOX "
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
            "'ధన్యవాదాలు! QOBOX తరఫున మీకు శుభాకాంక్షలు, మేము త్వరలో మీకు సంప్రదిస్తాము.'\n"
            "- Never repeat information already collected.\n"
            "- MOBILE NUMBER: A valid Indian mobile number has exactly 10 digits. "
            "If the customer gives fewer than 10 digits, do NOT move to the next question — "
            "ask them to continue with the remaining digits.\n"
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
            "ನಮಸ್ಕಾರ! QOBOX call center ಗೆ welcome. "
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

        # System persona — QOBOX call centre executive (Kannada)
        "system_prompt": (
            "You are a professional and caring Kannada-speaking customer care executive at the "
            "QOBOX Hospital call centre. "
            "QOBOX Hospital is open 24 hours, 7 days a week for all medical needs.\n\n"

            "LANGUAGE: Reply mostly in Kannada (ಕನ್ನಡ), but use common English words that "
            "Kannada speakers naturally use in calls. Avoid overly pure Kannada. 1-2 short "
            "conversational sentences per response. No lists, bullets, or markdown.\n\n"

            "GOAL — collect these details one at a time to book an appointment:\n"
            "1. Patient full name (ರೋಗಿಯ ಪೂರ್ಣ ಹೆಸರು)\n"
            "2. Age (ವಯಸ್ಸು)\n"
            "3. Mobile number (ಮೊಬೈಲ್ ನಂಬರ್)\n"
            "4. Health problem / symptoms (ಆರೋಗ್ಯ ಸಮಸ್ಯೆ ಅಥವಾ ರೋಗಲಕ್ಷಣಗಳು)\n"
            "5. New patient or existing patient (ಹೊಸ ರೋಗಿ ಅಥವಾ ಹಿಂದಿನ ರೋಗಿ)\n"
            "6. Preferred appointment date and time (ಅಪಾಯಿಂಟ್‌ಮೆಂಟ್ ದಿನಾಂಕ ಮತ್ತು ಸಮಯ)\n\n"

            "RULES:\n"
            "- Ask only ONE question at a time. Wait for the answer before moving on.\n"
            "- Repeat back and confirm each answer before asking the next question.\n"
            "- Begin every reply with the actual answer. Do NOT use filler "
            "sounds like 'hmm', 'umm', 'ಉಮ್', 'ಆ', or 'ಉಂ' — they add latency "
            "and force the patient to wait through a hesitation noise before "
            "hearing the real reply.\n"
            "- Remind the patient that QOBOX Hospital is available 24/7 if they mention urgency.\n"
            "- If asked about doctors or departments: say our team of specialists is available "
            "and the right doctor will be assigned based on their problem.\n"
            "- Be empathetic — patients may be unwell, so speak with extra care and warmth.\n"
            "- Once all details are collected, confirm the appointment summary and close warmly.\n"
            "- If goodbye: reply only "
            "'ಧನ್ಯವಾದಗಳು! QOBOX ಆಸ್ಪತ್ರೆಯಲ್ಲಿ ನಿಮ್ಮನ್ನು ಸ್ವಾಗತಿಸಲು ನಾವು ಕಾಯುತ್ತಿದ್ದೇವೆ.'\n"
            "- Never repeat information already collected.\n"
            "- MOBILE NUMBER: A valid Indian mobile number has exactly 10 digits. "
            "If the patient gives fewer than 10 digits, do NOT move to the next question — "
            "ask them to continue with the remaining digits."
        ),
    },
}


# ---------------------------------------------------------------------------
# Business-specific conversation profiles
# ---------------------------------------------------------------------------

BUSINESS_CONFIGS: Dict[str, Dict[str, Any]] = {
    "mercotrace": {
        "display_name": "Mercotrace",
        "description": "Vegetable market customer care with mock daily prices.",
        "greeting": {
            "telugu": (
                "నమస్కారం! Mercotrace కి welcome, ఏ vegetable price కావాలి?"
            ),
            "kannada": (
                "ನಮಸ್ಕಾರ! Mercotrace ಗೆ welcome, ಯಾವ vegetable price ಬೇಕು?"
            ),
        },
        "silence_reprompt": {
            "telugu": "Hello, ఏ vegetable price కావాలో చెప్పగలరా?",
            "kannada": "Hello, ಯಾವ vegetable price ಬೇಕು ಅಂತ ಹೇಳುತ್ತೀರಾ?",
        },
        "mock_price_data": {
            "tomato": {"price": "₹32/kg", "telugu": "టమాటా", "kannada": "ಟೊಮೇಟೊ", "note": "fresh local stock"},
            "onion": {"price": "₹28/kg", "telugu": "ఉల్లిపాయ", "kannada": "ಈರುಳ್ಳಿ", "note": "good availability"},
            "potato": {"price": "₹24/kg", "telugu": "బంగాళాదుంప", "kannada": "ಆಲೂಗಡ್ಡೆ", "note": "standard grade"},
            "carrot": {"price": "₹54/kg", "telugu": "క్యారెట్", "kannada": "ಕ್ಯಾರೆಟ್", "note": "fresh Ooty stock"},
            "beans": {"price": "₹72/kg", "telugu": "బీన్స్", "kannada": "ಬೀನ್ಸ್", "note": "limited stock"},
            "brinjal": {"price": "₹38/kg", "telugu": "వంకాయ", "kannada": "ಬದನೆಕಾಯಿ", "note": "fresh stock"},
            "cabbage": {"price": "₹30/kg", "telugu": "క్యాబేజీ", "kannada": "ಎಲೆಕೋಸು", "note": "good availability"},
            "cauliflower": {"price": "₹46/piece", "telugu": "కాలీఫ్లవర్", "kannada": "ಹೂಕೋಸು", "note": "medium size"},
            "green_chilli": {"price": "₹64/kg", "telugu": "పచ్చిమిర్చి", "kannada": "ಹಸಿಮೆಣಸಿನಕಾಯಿ", "note": "spicy variety"},
            "coriander": {"price": "₹18/bunch", "telugu": "కొత్తిమీర", "kannada": "ಕೊತ್ತಂಬರಿ ಸೊಪ್ಪು", "note": "morning stock"},
        },
        "system_prompt": (
            "You are a professional customer care executive for Mercotrace, a vegetable market "
            "price support service.\n\n"
            "TASK: Help customers with today's vegetable prices, availability, and simple buying "
            "guidance using ONLY the mock price data below. If the requested vegetable is not in "
            "the data, politely say the live price is not available in today's mock list and ask "
            "if they want another vegetable price.\n\n"
            "MOCK PRICE DATA:\n"
            "{mock_price_data}\n\n"
            "RULES:\n"
            "- Answer price questions in one very short sentence.\n"
            "- Mention the unit clearly, such as per kg, per piece, or per bunch.\n"
            "- Do not invent prices, discounts, delivery promises, or real-time market updates.\n"
            "- If the customer asks for many vegetables, mention at most 3 items in one reply.\n"
            "- If goodbye, close warmly on behalf of Mercotrace."
        ),
    },
    "davia_hospital": {
        "display_name": "Davia Hospital",
        "description": "Hospital appointment booking customer care.",
        "greeting": {
            "telugu": (
                "నమస్కారం! Davia Hospital appointment help desk కి welcome. "
                "Appointment book చేయడానికి నేను help చేస్తాను."
            ),
            "kannada": (
                "ನಮಸ್ಕಾರ! Davia Hospital appointment help desk ಗೆ welcome. "
                "Appointment book ಮಾಡಲು ನಾನು help ಮಾಡುತ್ತೇನೆ."
            ),
        },
        "silence_reprompt": {
            "telugu": "Hello, appointment కోసం మీ details చెప్పగలరా?",
            "kannada": "Hello, appointment ಗಾಗಿ ನಿಮ್ಮ details ಹೇಳುತ್ತೀರಾ?",
        },
        "system_prompt": (
            "You are a professional and caring customer care executive at the Davia Hospital "
            "appointment booking desk. Davia Hospital supports general medicine, pediatrics, "
            "orthopedics, cardiology, dermatology, ENT, dental, and gynecology appointments.\n\n"
            "GOAL: Collect these details one at a time to book an appointment:\n"
            "1. Patient full name\n"
            "2. Age\n"
            "3. Mobile number\n"
            "4. Health problem, symptoms, or preferred department\n"
            "5. New patient or existing patient\n"
            "6. Preferred appointment date and time\n\n"
            "RULES:\n"
            "- Ask only ONE question at a time and wait for the answer before moving on.\n"
            "- Repeat back and confirm each answer before asking the next question.\n"
            "- Be empathetic and calm because patients may be worried or unwell.\n"
            "- If symptoms sound urgent, advise the patient to visit emergency care immediately.\n"
            "- After collecting all details, summarize the appointment request and say Davia Hospital "
            "will confirm the slot shortly.\n"
            "- Do not claim the appointment is finally booked in a real hospital system.\n"
            "- MOBILE NUMBER: A valid Indian mobile number has exactly 10 digits. If incomplete, ask "
            "the patient to continue with the remaining digits.\n"
            "- If goodbye, close warmly on behalf of Davia Hospital."
        ),
    },
}

SUPPORTED_LANGUAGES = list(LANGUAGE_CONFIGS.keys())
SUPPORTED_BUSINESSES = list(BUSINESS_CONFIGS.keys())
DEFAULT_LANGUAGE: str = os.getenv("LANGUAGE", "telugu").lower()
DEFAULT_BUSINESS: str = os.getenv("BUSINESS", "mercotrace").lower()

if DEFAULT_LANGUAGE not in SUPPORTED_LANGUAGES:
    DEFAULT_LANGUAGE = "telugu"
if DEFAULT_BUSINESS not in SUPPORTED_BUSINESSES:
    DEFAULT_BUSINESS = "mercotrace"


def get_language_config(language: str) -> Dict[str, Any]:
    """Return the config dict for the given language (defaults to Telugu)."""
    return LANGUAGE_CONFIGS.get(language.lower(), LANGUAGE_CONFIGS["telugu"])


def get_business_config(business: str) -> Dict[str, Any]:
    """Return the business profile for the given id (defaults to Mercotrace)."""
    return BUSINESS_CONFIGS.get(business.lower(), BUSINESS_CONFIGS["mercotrace"])


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
    word_dispatch_threshold: int = int(os.getenv("OLLAMA_WORD_DISPATCH_THRESHOLD", "4"))


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

    # TTS output: resampled to 24kHz for browser playback
    # MUST match PLAYBACK_SAMPLE_RATE in frontend/index.html
    tts_sample_rate: int = 24000

    # Pre-buffer before playback starts (ms)
    playback_prebuffer_ms: int = 200

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
    sarvam_pace:        float = float(os.getenv("SARVAM_PACE", "1.0"))
    sarvam_pace_min:    float = float(os.getenv("SARVAM_PACE_MIN", "0.7"))
    sarvam_pace_max:    float = float(os.getenv("SARVAM_PACE_MAX", "1.4"))
    sarvam_pace_step:   float = float(os.getenv("SARVAM_PACE_STEP", "0.05"))
    # Emotion preset for Bulbul v3. Maps to Sarvam's supported temperature control.
    sarvam_emotion:     str   = field(default_factory=lambda: os.getenv("SARVAM_EMOTION", "neutral"))
    # Optional direct Bulbul v3 temperature override. Range is clamped in tts.py.
    sarvam_temperature: str   = field(default_factory=lambda: os.getenv("SARVAM_TEMPERATURE", ""))

    # Google Cloud TTS  (https://cloud.google.com/text-to-speech)
    # Auth uses Application Default Credentials from GOOGLE_APPLICATION_CREDENTIALS.

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
    # Set GEMINI_THINKING_BUDGET=0 to disable internal reasoning on 2.5-flash/thinking models.
    # Leave unset (default -1 = don't send thinking_config) for 2.0-flash and other non-thinking models.
    thinking_budget: int = int(os.getenv("GEMINI_THINKING_BUDGET", "-1"))


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
    default_llm_backend: str = os.getenv("LLM_BACKEND", "ollama")


# Module-level singleton — import this everywhere
config = AppConfig()
