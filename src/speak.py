# import os
import platform
import subprocess
from TTS.api import TTS

# Preload Coqui TTS for non-macOS fallback
tts = TTS(
    model_name="tts_models/en/ljspeech/glow-tts",
    progress_bar=False,
    gpu=False
)

# Map macOS voices
_SIRI_VOICE = "Alex"

def speak(text: str):
    """
    Speak `text` aloud.
    - On macOS: uses `say` with a Siri-style voice.
    - Elsewhere: uses Coqui TTS as a fallback.
    """
    system = platform.system()
    if system == "Darwin":
        # macOS built-in TTS using subprocess to avoid shell issues
        subprocess.run(["say", "-v", _SIRI_VOICE, text])
    else:
        # fallback for Linux/Windows
        out = "recap_speech.wav"
        tts.tts_to_file(text=text, file_path=out)
        if system == "Linux":
            subprocess.run(["aplay", out])
        elif system == "Windows":
            subprocess.run([
                "powershell",
                "-c",
                f'(New-Object Media.SoundPlayer "{out}").PlaySync()'
            ])
        else:
            subprocess.run(["play", out])  # requires sox
