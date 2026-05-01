"""Synthesize text to output.mp3 using Google Cloud Text-to-Speech.

Authentication is handled by the Google SDK through
GOOGLE_APPLICATION_CREDENTIALS. Do not pass or store an API key.
"""

import argparse
import os
from pathlib import Path

from google.cloud import texttospeech


def synthesize_to_mp3(
    text: str,
    output_path: str = "output.mp3",
    language_code: str = "en-US",
    voice_name: str | None = None,
) -> None:
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not credentials_path:
        raise RuntimeError("Set GOOGLE_APPLICATION_CREDENTIALS to your service account JSON file.")
    if not Path(credentials_path).expanduser().is_file():
        raise RuntimeError(f"GOOGLE_APPLICATION_CREDENTIALS file not found: {credentials_path}")

    client = texttospeech.TextToSpeechClient()

    voice_kwargs = {"language_code": language_code}
    if voice_name:
        voice_kwargs["name"] = voice_name
    else:
        voice_kwargs["ssml_gender"] = texttospeech.SsmlVoiceGender.NEUTRAL

    response = client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=text),
        voice=texttospeech.VoiceSelectionParams(**voice_kwargs),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
        ),
    )

    Path(output_path).write_bytes(response.audio_content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MP3 audio with Google Cloud TTS.")
    parser.add_argument("text", nargs="?", default="Hello world", help="Text to synthesize.")
    parser.add_argument("--output", default="output.mp3", help="Output MP3 path.")
    parser.add_argument("--language-code", default="en-US", help="Google TTS language code.")
    parser.add_argument("--voice", default=None, help="Optional Google TTS voice name.")
    args = parser.parse_args()

    synthesize_to_mp3(
        text=args.text,
        output_path=args.output,
        language_code=args.language_code,
        voice_name=args.voice,
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
