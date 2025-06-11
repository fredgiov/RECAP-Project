# speak.py

import platform
import subprocess
import tempfile
import torch
from TTS.api import TTS

# 1) Choose Coqui’s best multilingual model
MODEL_NAME = "tts_models/multilingual/multi-dataset/vits"
tts = TTS(
    model_name=MODEL_NAME,
    progress_bar=False,
    gpu=torch.cuda.is_available()  # use GPU if available
)

def speak(text: str, speaker: str = None, language: str = None):
    """
    Speak `text` aloud using Coqui’s multilingual VITS model.
    
    - speaker: one of tts.speakers (e.g. 'en_0', 'de_0', ...)
    - language: one of tts.languages (e.g. 'en', 'de', 'fr', ...)
    
    If you don’t pass speaker or language, it will default
    to the first available in each list.
    """
    # Pick defaults
    if speaker is None:
        speaker = tts.speakers[0]
    if language is None:
        language = tts.languages[0]

    # Generate a temp WAV
    out_path = tempfile.mktemp(suffix=".wav")
    tts.tts_to_file(
        text=text,
        speaker=speaker,
        language=language,
        file_path=out_path
    )

    # Play it back
    system = platform.system()
    if system == "Windows":
        subprocess.run([
            "powershell", "-c",
            f'(New-Object Media.SoundPlayer "{out_path}").PlaySync()'
        ], check=True)
    elif system == "Darwin":
        subprocess.run(["afplay", out_path], check=True)
    else:
        subprocess.run(["aplay", out_path], check=True)

# --- optional helper to see what you can choose ---
if __name__ == "__main__":
    print("Available languages:", tts.languages)
    print("Available speakers:", tts.speakers)
