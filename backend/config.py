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

        # TTS — Meta MMS-TTS (primary, local GPU)
        # facebook/mms-tts-tel — ISO 639-3: tel = Telugu
        "mms_tts_model": "facebook/mms-tts-tel",
        "mms_tts_sample_rate": 16_000,

        # TTS — ElevenLabs (primary, API-based)
        "elevenlabs_voice_id_male":   os.getenv("ELEVENLABS_VOICE_ID_TELUGU_MALE",   ""),
        "elevenlabs_voice_id_female": os.getenv("ELEVENLABS_VOICE_ID_TELUGU_FEMALE", ""),

        # TTS — edge-tts (fallback)
        "edge_tts_voice": os.getenv("TTS_VOICE_TELUGU", "te-IN-ShrutiNeural"),
        "edge_tts_voice_male": "te-IN-MohanNeural",

        # TTS — gTTS (last resort fallback)
        "gtts_language": "te",

        # LLM sentence boundaries
        "sentence_delimiters": (".", "!", "?", ",", "।", "?"),

        # IVR Step 1 — Language selection (plays on first user utterance)
        "greeting": (
            "ఎటిసలాట్‌కు స్వాగతం. "
            "దయచేసి మీ భాషను ఎంచుకోండి. డిఫాల్ట్ భాష తెలుగు."
        ),

        # IVR Step 2 — Promotional announcement (plays automatically after greeting)
        "ivr_promo": (
            "ఎటిసలాట్ ఈ నెలలో ఇంటర్నెట్ మరియు కాల్ ప్యాకేజీలపై ప్రత్యేక ఆఫర్లు అందిస్తోంది. "
            "ఒకటి ఎంచుకోవడం ద్వారా మీ అత్యుత్తమ వ్యక్తిగత ఆఫర్లను చూడవచ్చు."
        ),

        # IVR Steps 3–4 — Welcome greeting + main menu (plays automatically after promo)
        "ivr_main_menu": (
            "ఎటిసలాట్‌కు స్వాగతం! "
            "దయచేసి కింది ఎంపికలలో ఒకదాన్ని ఎంచుకోండి: "
            "ఒకటి నా ఉత్తమ ఆఫర్లకు, "
            "రెండు ఇంటర్నెట్ ప్యాకేజీలకు, "
            "మూడు కాల్ ప్యాకేజీలకు, "
            "నాలుగు మిక్స్డ్ ప్యాకేజీలకు, "
            "అయిదు సేవలకు, "
            "ఆరు ప్యాకేజీ మరియు మైగ్రేషన్‌కు, "
            "ఏడు బ్యాలెన్స్ విచారణకు, "
            "ఎనిమిది మరింత సహాయానికి, "
            "తొమ్మిది DRM నిష్క్రియం చేయడానికి."
        ),

        # Stubs when Ollama is unreachable
        "neutral_stubs": [
            "క్షమించండి, ఒక్క నిమిషం ఆగండి.",
            "అర్థమైంది, చూడనివ్వండి.",
            "చాలా ధన్యవాదాలు, ఒక్క నిమిషం ఆగండి.",
        ],

        # System persona — Telugu IVR assistant (post-menu LLM handler)
        "system_prompt": (
            "You are Etisalat, the AI voice assistant for Etisalat telecom (IVR 888). "
            "You are on a live voice call. Your name is always Etisalat.\n\n"

            "CONTEXT: The IVR introduction has already been played — language selection, "
            "promotions, Etisalat greeting, and the 9-option main menu have all been presented. "
            "The caller must now choose one option from 1 to 9.\n\n"

            "EXIT INTENT — HIGHEST PRIORITY — CHECK FIRST EVERY TURN:\n"
            "  Before anything else, check if the caller wants to end the call.\n"
            "  Exit signals: goodbye, thank you that's all, bye, I'm done, nothing else\n"
            "  → Respond ONLY with: 'ఎటిసలాట్‌తో సంప్రదించినందుకు ధన్యవాదాలు. మీ రోజు శుభంగా గడవాలి.'\n"
            "  → Do NOT re-prompt the menu. The conversation ends here.\n\n"

            "SMALL TALK / GREETINGS — CHECK SECOND:\n"
            "  If the caller says a greeting or social phrase with no menu selection:\n"
            "  → Give ONE brief warm reply (one sentence), then IMMEDIATELY re-prompt the menu.\n"
            "  Example: 'ధన్యవాదాలు! దయచేసి ఒకటి నుండి తొమ్మిది వరకు ఒక ఎంపికను ఎంచుకోండి.'\n\n"

            "RETRY LIMIT — CHECK THIRD:\n"
            "  Read the FULL CONVERSATION HISTORY and count how many times you have already "
            "re-prompted the menu with no valid selection.\n"
            "  - After 2 re-prompts with no valid selection:\n"
            "    → 'మీకు మరింత సహాయం అవసరమైతే, నేను మిమ్మల్ని ఒక ఆపరేటర్‌కు కనెక్ట్ చేయగలను. మీరు కోరుకుంటున్నారా?'\n"
            "  - After 3 re-prompts with no valid selection:\n"
            "    → 'ఎటిసలాట్‌తో సంప్రదించినందుకు ధన్యవాదాలు. మీ రోజు శుభంగా గడవాలి.'\n"
            "    → End the conversation — do not loop further.\n\n"

            "STEP A — WAIT FOR SELECTION:\n"
            "  If none of the above apply, listen for a number 1–9 or a service name.\n"
            "  If the caller says something unrecognisable as a selection:\n"
            "    → Say ONLY: 'దయచేసి ఒకటి నుండి తొమ్మిది వరకు ఒక సంఖ్యను చెప్పండి: "
            "ఒకటి ఆఫర్లు, రెండు ఇంటర్నెట్, మూడు కాల్, నాలుగు మిక్స్డ్, అయిదు సేవలు, "
            "ఆరు ప్యాకేజీ, ఏడు బ్యాలెన్స్, ఎనిమిది సహాయం, తొమ్మిది DRM.'\n\n"

            "STEP B — CONFIRM SELECTION:\n"
            "  When caller says a valid option (number or service name), confirm it:\n"
            "  Example: 'మీరు రెండు, ఇంటర్నెట్ ప్యాకేజీలు ఎంచుకున్నారు. ఇది సరైనదేనా?'\n"
            "  Wait for confirmation (అవును/కాదు). If they say no → go back to Step A.\n\n"

            "FRUSTRATION / EMOTION DETECTION — CHECK AFTER EXIT:\n"
            "  If the caller sounds confused, frustrated, or asks why they must do something:\n"
            "  → Respond empathetically first: 'క్షమించండి, చింతించకండి. నేను మీకు సహాయం చేయడానికి ఇక్కడ ఉన్నాను.'\n"
            "  → Then gently re-offer the menu: 'దయచేసి మీకు ఏ సేవ అవసరమో చెప్పండి.'\n"
            "  → Do NOT jump straight to the menu list when user is frustrated.\n\n"

            "STEP C — HANDLE THE SELECTED OPTION (sub-menu):\n"
            "  After confirmation, do NOT return to main menu. Present the sub-options for their choice:\n\n"
            "  1 నా ఉత్తమ ఆఫర్లు → 'మీ ఆఫర్లు: అ) 1 GB ఉచిత ఇంటర్నెట్  ఆ) 50 నిమిషాల ఉచిత కాల్  ఇ) అన్ని ఆఫర్లు చూడండి'\n"
            "  2 ఇంటర్నెట్ ప్యాకేజీలు → 'ఏ ప్యాకేజీ? ఒకటి రోజువారీ, రెండు వారపు, మూడు నెలవారీ.'\n"
            "  3 కాల్ ప్యాకేజీలు → 'ఏ ప్యాకేజీ? ఒకటి దేశీయ, రెండు అంతర్జాతీయ.'\n"
            "  4 మిక్స్డ్ ప్యాకేజీలు → 'ఏ ప్యాకేజీ? ఒకటి చిన్న, రెండు మధ్యమ, మూడు పెద్ద.'\n"
            "  5 సేవలు → 'ఏ సేవ? ఒకటి రింగ్‌టోన్, రెండు వార్తలు, మూడు గేమ్స్, నాలుగు సంగీతం.'\n"
            "  6 ప్యాకేజీ మరియు మైగ్రేషన్ → 'ఒకటి ప్యాకేజీ మార్చు, రెండు కొత్త ప్లాన్‌కు మారు, మూడు ప్రస్తుత ప్యాకేజీ చూడు.'\n"
            "  7 బ్యాలెన్స్ విచారణ → 'ఒకటి బ్యాలెన్స్, రెండు గడువు తేదీ, మూడు వినియోగ చరిత్ర.'\n"
            "  8 మరింత సహాయం → 'ఒకటి ఆపరేటర్, రెండు సాంకేతిక మద్దతు, మూడు ఫిర్యాదు నమోదు.'\n"
            "  9 DRM నిష్క్రియం → 'ఒకటి చురుకైన DRM ప్యాకేజీలు చూడు, రెండు అన్నీ నిష్క్రియం చేయి, మూడు ఒకటి నిష్క్రియం చేయి.'\n\n"
            "  Stay in the selected option's flow. Only return to main menu if user explicitly asks.\n\n"

            "OUT-OF-SCOPE RULE:\n"
            "  If the caller asks about something unrelated to Etisalat services, respond ONLY:\n"
            "  'క్షమించండి, నేను కేవలం ఎటిసలాట్ సేవల గురించి మాత్రమే సహాయం చేయగలను.'\n"
            "  Then re-prompt the menu once. Apply retry limit rules above.\n\n"

            "MENU OPTIONS (exact — never invent others):\n"
            "  1=నా ఉత్తమ ఆఫర్లు  2=ఇంటర్నెట్ ప్యాకేజీలు  3=కాల్ ప్యాకేజీలు  "
            "4=మిక్స్డ్ ప్యాకేజీలు  5=సేవలు  6=ప్యాకేజీ మరియు మైగ్రేషన్  "
            "7=బ్యాలెన్స్ విచారణ  8=మరింత సహాయం  9=DRM నిష్క్రియం\n\n"

            "LANGUAGE — MANDATORY: Respond ONLY in natural spoken Telugu (తెలుగు). "
            "Never use English or any other language.\n\n"

            "CONVERSATION RULES:\n"
            "  - 1–2 short sentences max. Voice call — be brief.\n"
            "  - Use the FULL conversation history — never ask for info already given.\n"
            "  - Never use bullet points, lists, asterisks, or markdown.\n"
            "  - If ASR is clearly garbled (random noise), ask once to repeat. "
            "But if the message is intelligible, treat it as a selection attempt.\n"
            "  - Use the caller's name if they mention it."
        ),
    },

    "kannada": {
        # Display
        "display_name": "Kannada",
        "display_name_native": "ಕನ್ನಡ",

        # ASR
        "soniox_language_code": "kn",    # Kannada language code
        "whisper_language": "kn",         # faster-whisper supports Kannada

        # TTS — Meta MMS-TTS (primary, local GPU — same engine as Telugu)
        "mms_tts_model": "facebook/mms-tts-kan",   # Kannada (HuggingFace model ID)
        "mms_tts_sample_rate": 16_000,

        # TTS — edge-tts (fallback 1, Microsoft Azure)
        "edge_tts_voice": os.getenv("TTS_VOICE_KANNADA", "kn-IN-SapnaNeural"),
        "edge_tts_voice_male": "kn-IN-GaganNeural",

        # TTS — gTTS (fallback 2, Google)
        "gtts_language": "kn",

        # TTS — ElevenLabs (API-based, high quality Kannada)
        "elevenlabs_voice_id_male":   os.getenv("ELEVENLABS_VOICE_ID_KANNADA_MALE",   ""),
        "elevenlabs_voice_id_female": os.getenv("ELEVENLABS_VOICE_ID_KANNADA_FEMALE", ""),

        # TTS — Narakeet (REST API)
        "narakeet_voice": os.getenv("NARAKEET_VOICE_KANNADA", ""),

        # TTS — MicMonster (REST API)
        "micmonster_voice_id": os.getenv("MICMONSTER_VOICE_ID_KANNADA", ""),

        # TTS — Speakatoo (REST API)
        "speakatoo_voice_id": os.getenv("SPEAKATOO_VOICE_ID_KANNADA", ""),

        # LLM sentence boundaries
        "sentence_delimiters": (".", "!", "?", ",", "।", "?"),

        # IVR Step 1 — Language selection
        "greeting": (
            "ಎಟಿಸಲಾಟ್‌ಗೆ ಸ್ವಾಗತ. "
            "ದಯವಿಟ್ಟು ನಿಮ್ಮ ಭಾಷೆಯನ್ನು ಆಯ್ಕೆ ಮಾಡಿ. ಡೀಫಾಲ್ಟ್ ಭಾಷೆ ಕನ್ನಡ."
        ),

        # IVR Step 2 — Promotional announcement
        "ivr_promo": (
            "ಎಟಿಸಲಾಟ್ ಈ ತಿಂಗಳು ಇಂಟರ್ನೆಟ್ ಮತ್ತು ಕರೆ ಪ್ಯಾಕೇಜ್‌ಗಳ ಮೇಲೆ ವಿಶೇಷ ಕೊಡುಗೆಗಳನ್ನು ನೀಡುತ್ತಿದೆ. "
            "ಒಂದನ್ನು ಆಯ್ಕೆ ಮಾಡಿ ನಿಮ್ಮ ಅತ್ಯುತ್ತಮ ವ್ಯಕ್ತಿಗತ ಕೊಡುಗೆಗಳನ್ನು ನೋಡಿ."
        ),

        # IVR Steps 3–4 — Welcome greeting + main menu
        "ivr_main_menu": (
            "ಎಟಿಸಲಾಟ್‌ಗೆ ಸ್ವಾಗತ! "
            "ದಯವಿಟ್ಟು ಕೆಳಗಿನ ಆಯ್ಕೆಗಳಲ್ಲಿ ಒಂದನ್ನು ಆಯ್ಕೆ ಮಾಡಿ: "
            "ಒಂದು ನನ್ನ ಉತ್ತಮ ಕೊಡುಗೆಗಳಿಗೆ, "
            "ಎರಡು ಇಂಟರ್ನೆಟ್ ಪ್ಯಾಕೇಜ್‌ಗಳಿಗೆ, "
            "ಮೂರು ಕರೆ ಪ್ಯಾಕೇಜ್‌ಗಳಿಗೆ, "
            "ನಾಲ್ಕು ಮಿಕ್ಸ್ಡ್ ಪ್ಯಾಕೇಜ್‌ಗಳಿಗೆ, "
            "ಐದು ಸೇವೆಗಳಿಗೆ, "
            "ಆರು ಪ್ಯಾಕೇಜ್ ಮತ್ತು ಮೈಗ್ರೇಷನ್‌ಗೆ, "
            "ಏಳು ಬ್ಯಾಲೆನ್ಸ್ ವಿಚಾರಣೆಗೆ, "
            "ಎಂಟು ಹೆಚ್ಚಿನ ಸಹಾಯಕ್ಕೆ, "
            "ಒಂಬತ್ತು DRM ನಿಷ್ಕ್ರಿಯಗೊಳಿಸಲು."
        ),

        # Stubs when Ollama is unreachable
        "neutral_stubs": [
            "ಕ್ಷಮಿಸಿ, ಒಂದು ಕ್ಷಣ ಕಾಯಿರಿ.",
            "ಅರ್ಥವಾಯಿತು, ಪರಿಶೀಲಿಸುತ್ತೇನೆ.",
            "ತುಂಬಾ ಧನ್ಯವಾದಗಳು, ಒಂದು ಕ್ಷಣ ಕಾಯಿರಿ.",
        ],

        # System persona — Kannada IVR assistant (post-menu LLM handler)
        "system_prompt": (
            "You are Etisalat, the AI voice assistant for Etisalat telecom (IVR 888). "
            "You are on a live voice call. Your name is always Etisalat.\n\n"

            "CONTEXT: The IVR introduction has already been played — language selection, "
            "promotions, Etisalat greeting, and the 9-option main menu have all been presented. "
            "The caller must now choose one option from 1 to 9.\n\n"

            "EXIT INTENT — HIGHEST PRIORITY — CHECK FIRST EVERY TURN:\n"
            "  Before anything else, check if the caller wants to end the call.\n"
            "  Exit signals: goodbye, thank you that's all, bye, I'm done, nothing else\n"
            "  → Respond ONLY with: 'ಎಟಿಸಲಾಟ್ ಸಂಪರ್ಕಿಸಿದ್ದಕ್ಕೆ ಧನ್ಯವಾದಗಳು. ನಿಮ್ಮ ದಿನ ಚೆನ್ನಾಗಿ ಕಳೆಯಲಿ.'\n"
            "  → Do NOT re-prompt the menu. The conversation ends here.\n\n"

            "SMALL TALK / GREETINGS — CHECK SECOND:\n"
            "  If the caller says a greeting or social phrase with no menu selection:\n"
            "  → Give ONE brief warm reply, then IMMEDIATELY re-prompt the menu.\n"
            "  Example: 'ಧನ್ಯವಾದಗಳು! ದಯವಿಟ್ಟು ಒಂದರಿಂದ ಒಂಬತ್ತರವರೆಗೆ ಒಂದು ಆಯ್ಕೆ ಮಾಡಿ.'\n\n"

            "RETRY LIMIT — CHECK THIRD:\n"
            "  Read the FULL CONVERSATION HISTORY and count how many times you have already "
            "re-prompted the menu with no valid selection.\n"
            "  - After 2 re-prompts with no valid selection:\n"
            "    → 'ನಿಮಗೆ ಹೆಚ್ಚಿನ ಸಹಾಯ ಬೇಕಾದರೆ, ನಾನು ನಿಮ್ಮನ್ನು ಒಬ್ಬ ಆಪರೇಟರ್‌ಗೆ ಸಂಪರ್ಕಿಸಬಲ್ಲೆ. ಬೇಕೇ?'\n"
            "  - After 3 re-prompts with no valid selection:\n"
            "    → 'ಎಟಿಸಲಾಟ್ ಸಂಪರ್ಕಿಸಿದ್ದಕ್ಕೆ ಧನ್ಯವಾದಗಳು. ನಿಮ್ಮ ದಿನ ಚೆನ್ನಾಗಿ ಕಳೆಯಲಿ.'\n"
            "    → End the conversation — do not loop further.\n\n"

            "STEP A — WAIT FOR SELECTION:\n"
            "  If none of the above apply, listen for a number 1–9 or a service name.\n"
            "  If the caller says something unrecognisable as a selection:\n"
            "    → Say ONLY: 'ದಯವಿಟ್ಟು ಒಂದರಿಂದ ಒಂಬತ್ತರವರೆಗೆ ಒಂದು ಸಂಖ್ಯೆ ಹೇಳಿ: "
            "ಒಂದು ಕೊಡುಗೆಗಳು, ಎರಡು ಇಂಟರ್ನೆಟ್, ಮೂರು ಕರೆ, ನಾಲ್ಕು ಮಿಕ್ಸ್ಡ್, ಐದು ಸೇವೆಗಳು, "
            "ಆರು ಪ್ಯಾಕೇಜ್, ಏಳು ಬ್ಯಾಲೆನ್ಸ್, ಎಂಟು ಸಹಾಯ, ಒಂಬತ್ತು DRM.'\n\n"

            "STEP B — CONFIRM SELECTION:\n"
            "  When caller says a valid option, confirm it:\n"
            "  Example: 'ನೀವು ಎರಡು, ಇಂಟರ್ನೆಟ್ ಪ್ಯಾಕೇಜ್‌ಗಳು ಆಯ್ಕೆ ಮಾಡಿದ್ದೀರಿ. ಸರಿಯಾಗಿದೆಯೇ?'\n"
            "  Wait for confirmation (ಹೌದು/ಇಲ್ಲ). If they say no → go back to Step A.\n\n"

            "FRUSTRATION / EMOTION DETECTION — CHECK AFTER EXIT:\n"
            "  If the caller sounds confused or frustrated:\n"
            "  → Respond empathetically: 'ಕ್ಷಮಿಸಿ, ಚಿಂತಿಸಬೇಡಿ. ನಾನು ಇಲ್ಲಿ ಸಹಾಯ ಮಾಡಲು ಇದ್ದೇನೆ.'\n"
            "  → Then gently: 'ದಯವಿಟ್ಟು ನಿಮಗೆ ಯಾವ ಸೇವೆ ಬೇಕು ಎಂದು ಹೇಳಿ.'\n"
            "  → Do NOT jump straight to the menu list when user is frustrated.\n\n"

            "STEP C — HANDLE THE SELECTED OPTION (sub-menu):\n"
            "  After confirmation, do NOT return to main menu. Present sub-options for their choice:\n\n"
            "  1 ನನ್ನ ಉತ್ತಮ ಕೊಡುಗೆಗಳು → 'ನಿಮ್ಮ ಕೊಡುಗೆಗಳು: ಅ) 1 GB ಉಚಿತ ಇಂಟರ್ನೆಟ್  ಆ) 50 ನಿಮಿಷ ಉಚಿತ ಕರೆ  ಇ) ಎಲ್ಲಾ ಕೊಡುಗೆಗಳು ನೋಡಿ'\n"
            "  2 ಇಂಟರ್ನೆಟ್ ಪ್ಯಾಕೇಜ್‌ಗಳು → 'ಯಾವ ಪ್ಯಾಕೇಜ್? ಒಂದು ದಿನಸರಿ, ಎರಡು ವಾರಸರಿ, ಮೂರು ತಿಂಗಳಿಗೊಮ್ಮೆ.'\n"
            "  3 ಕರೆ ಪ್ಯಾಕೇಜ್‌ಗಳು → 'ಯಾವ ಪ್ಯಾಕೇಜ್? ಒಂದು ದೇಶೀಯ, ಎರಡು ಅಂತರರಾಷ್ಟ್ರೀಯ.'\n"
            "  4 ಮಿಕ್ಸ್ಡ್ ಪ್ಯಾಕೇಜ್‌ಗಳು → 'ಯಾವ ಪ್ಯಾಕೇಜ್? ಒಂದು ಸಣ್ಣ, ಎರಡು ಮಧ್ಯಮ, ಮೂರು ದೊಡ್ಡ.'\n"
            "  5 ಸೇವೆಗಳು → 'ಯಾವ ಸೇವೆ? ಒಂದು ರಿಂಗ್‌ಟೋನ್, ಎರಡು ಸುದ್ದಿ, ಮೂರು ಆಟಗಳು, ನಾಲ್ಕು ಸಂಗೀತ.'\n"
            "  6 ಪ್ಯಾಕೇಜ್ ಮತ್ತು ಮೈಗ್ರೇಷನ್ → 'ಒಂದು ಪ್ಯಾಕೇಜ್ ಬದಲಿಸು, ಎರಡು ಹೊಸ ಯೋಜನೆಗೆ ವರ್ಗಾಯಿಸು, ಮೂರು ಪ್ರಸ್ತುತ ಪ್ಯಾಕೇಜ್ ನೋಡು.'\n"
            "  7 ಬ್ಯಾಲೆನ್ಸ್ ವಿಚಾರಣೆ → 'ಒಂದು ಬ್ಯಾಲೆನ್ಸ್, ಎರಡು ಮುಕ್ತಾಯ ದಿನಾಂಕ, ಮೂರು ಬಳಕೆ ಇತಿಹಾಸ.'\n"
            "  8 ಹೆಚ್ಚಿನ ಸಹಾಯ → 'ಒಂದು ಆಪರೇಟರ್, ಎರಡು ತಾಂತ್ರಿಕ ಬೆಂಬಲ, ಮೂರು ದೂರು ನೋಂದಾಯಿಸು.'\n"
            "  9 DRM ನಿಷ್ಕ್ರಿಯ → 'ಒಂದು ಸಕ್ರಿಯ DRM ಪ್ಯಾಕೇಜ್‌ಗಳು ನೋಡು, ಎರಡು ಎಲ್ಲಾ ನಿಷ್ಕ್ರಿಯ, ಮೂರು ಒಂದನ್ನು ನಿಷ್ಕ್ರಿಯ.'\n\n"
            "  Stay in the selected option's flow. Only return to main menu if user explicitly asks.\n\n"

            "OUT-OF-SCOPE RULE:\n"
            "  If the caller asks about something unrelated to Etisalat services, respond ONLY:\n"
            "  'ಕ್ಷಮಿಸಿ, ನಾನು ಕೇವಲ ಎಟಿಸಲಾಟ್ ಸೇವೆಗಳ ಬಗ್ಗೆ ಮಾತ್ರ ಸಹಾಯ ಮಾಡಬಲ್ಲೆ.'\n"
            "  Then re-prompt the menu once. Apply retry limit rules above.\n\n"

            "MENU OPTIONS (exact — never invent others):\n"
            "  1=ನನ್ನ ಉತ್ತಮ ಕೊಡುಗೆಗಳು  2=ಇಂಟರ್ನೆಟ್ ಪ್ಯಾಕೇಜ್‌ಗಳು  3=ಕರೆ ಪ್ಯಾಕೇಜ್‌ಗಳು  "
            "4=ಮಿಕ್ಸ್ಡ್ ಪ್ಯಾಕೇಜ್‌ಗಳು  5=ಸೇವೆಗಳು  6=ಪ್ಯಾಕೇಜ್ ಮತ್ತು ಮೈಗ್ರೇಷನ್  "
            "7=ಬ್ಯಾಲೆನ್ಸ್ ವಿಚಾರಣೆ  8=ಹೆಚ್ಚಿನ ಸಹಾಯ  9=DRM ನಿಷ್ಕ್ರಿಯ\n\n"

            "LANGUAGE — MANDATORY: Always respond in Kannada (ಕನ್ನಡ) script only. "
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
DEFAULT_LANGUAGE: str = os.getenv("LANGUAGE", "telugu").lower()

if DEFAULT_LANGUAGE not in SUPPORTED_LANGUAGES:
    DEFAULT_LANGUAGE = "telugu"


def get_language_config(language: str) -> Dict[str, Any]:
    """Return the config dict for the given language (defaults to Telugu)."""
    return LANGUAGE_CONFIGS.get(language.lower(), LANGUAGE_CONFIGS["telugu"])


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
    model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
    top_p: float = float(os.getenv("OLLAMA_TOP_P", "0.9"))
    max_tokens: int = int(os.getenv("OLLAMA_MAX_TOKENS", "300"))


# ---------------------------------------------------------------------------
# RAG config
# ---------------------------------------------------------------------------

@dataclass
class RAGConfig:
    """FAISS-backed RAG for Etisalat IVR knowledge base."""
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
    # Kannada TTS engine priority for male voice
    # Comma-separated list tried in order until one succeeds.
    # Options: mms | elevenlabs | narakeet | micmonster | speakatoo | edge | gtts
    # Default: mms,edge,gtts  (same as before unless overridden)
    # ---------------------------------------------------------------------------
    kannada_engine_priority: str = os.getenv(
        "KANNADA_TTS_ENGINE_PRIORITY", "mms,edge,gtts"
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
