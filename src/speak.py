import os
import platform
from TTS.api import TTS

# Preload Coqui TTS for non-macOS fallback
tts = TTS(
    model_name="tts_models/en/ljspeech/glow-tts",
    progress_bar=False,
    gpu=False
)

# Map macOS voices
_SIRI_VOICE = "Samantha"  # or try "Siri" / "Victoria" / "Alex" etc.

def speak(text: str):
    """
    Speak `text` aloud.
    - On macOS: uses `say` with a Siri-style voice.
    - Elsewhere: uses Coqui TTS as a fallback.
    """
    system = platform.system()
    if system == "Darwin":
        # macOS built-in TTS
        # you can list available voices via: `say -v ?`
        os.system(f'say -v "{_SIRI_VOICE}" {text!r}')
    else:
        # fallback for Linux/Windows
        out = "recap_speech.wav"
        tts.tts_to_file(text=text, file_path=out)
        if system == "Linux":
            os.system(f"aplay {out}")
        elif system == "Windows":
            os.system(f'powershell -c (New-Object Media.SoundPlayer "{out}").PlaySync()')
        else:
            # last resort
            os.system(f"play {out}")  # requires sox
