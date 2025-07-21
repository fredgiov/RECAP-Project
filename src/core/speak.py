# speak.py

import os
import platform
import boto3
import sounddevice as sd
import numpy as np

# ─── 1) macOS guard ─────────────────────────────────────────────────────────
if platform.system() != "Darwin":
    raise OSError("This TTS helper is macOS-specific; please run on a Mac.")

# ─── 2) AWS credentials ─────────────────────────────────────────────────────
# boto3 will pick up AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
# and AWS_DEFAULT_REGION from env or ~/.aws/credentials
polly = boto3.client(
    "polly",
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
)

# ─── 3) Map ISO language codes to Polly VoiceIds ────────────────────────────
VOICE_MAP = {
    "en":    "Joanna",    # US English
    "en-gb": "Amy",       # British English
    "es":    "Lucia",     # Castilian Spanish
    "es-us": "Lupe",      # Latin American Spanish
    "fr":    "Mathieu",   # French
    "de":    "Vicki",     # German
    "it":    "Carla",     # Italian
    "pt":    "Vitoria",   # Portuguese (pt-PT)
    "ja":    "Mizuki",    # Japanese
    "ko":    "Seoyeon",   # Korean
    "zh":    "Zhiyu",     # Mandarin Chinese
    "hi":    "Aditi",     # Hindi
    "ar":    "Zeina",     # Arabic
    "nl":    "Lotte",     # Dutch
    "pl":    "Ewa",       # Polish
    "tr":    "Filiz",     # Turkish
    # …add more as needed
}

def _normalize_lang(code: str) -> str:
    """
    Collapse codes like “en-US” → “en” and pick a VoiceId.
    Falls back to “Joanna” (US English).
    """
    base = code.lower().split("-")[0]
    return VOICE_MAP.get(base, VOICE_MAP["en"])

# ─── 4) speak() ─────────────────────────────────────────────────────────────
def speak(text: str, play: bool, language: str = "en") -> None:
    """
    Synthesize `text` via Amazon Polly and play it.
    """
    voice_id = _normalize_lang(language)
    resp = polly.synthesize_speech(
        Text=text,
        OutputFormat="pcm",
        VoiceId=voice_id,
    )
    stream = resp.get("AudioStream")
    if not stream:
        print(f"[Error] Polly returned no audio (voice={voice_id}).")
        return

    pcm = stream.read()
    audio = np.frombuffer(pcm, dtype=np.int16)
    if play and audio.size > 0:
        sd.play(audio, samplerate=16000)
        sd.wait()
