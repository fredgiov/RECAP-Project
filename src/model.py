import os
import time
import tempfile
import platform

import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
import whisper
import ollama
from colorama import Fore, Style, init
from speak import speak

init(autoreset=True)

# ————————————————————————————————————————————————————————————————————————————————
# 0) Determine best device: Jetson Nano -> cuda, Mac -> mps, else cpu
arch = platform.machine()
if arch == "aarch64" and torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"{Fore.CYAN}Loading Whisper medium.en on CPU to patch sparse weights…{Style.RESET_ALL}")
asr = whisper.load_model("medium.en", device="cpu")

# Patch any sparse buffers before moving to our target device
for name, buf in list(asr.named_buffers()):
    if buf.layout == torch.sparse_coo:
        dense = buf.to_dense()
        asr.register_buffer(name, dense)

print(f"{Fore.CYAN}Moving Whisper model to {device.upper()}…{Style.RESET_ALL}")
asr = asr.to(device)

# ————————————————————————————————————————————————————————————————————————————————
# Detect if a microphone is available
try:
    input_devices = [d for d in sd.query_devices() if d['max_input_channels'] > 0]
    has_mic = len(input_devices) > 0
except Exception:
    has_mic = False

if not has_mic:
    print(f"{Fore.YELLOW}[Warning] No microphone detected. Falling back to text input.{Style.RESET_ALL}")

# ————————————————————————————————————————————————————————————————————————————————
def get_voice_input(
    timeout: float = 255.0,        # max recording length (seconds)
    silence_duration: float = 2.5, # stop after this much continuous quiet
    fs: int = 16_000,              # sample rate
    chunk_ms: int = 30,            # how big each frame is for silence checking
    threshold: float = 500.0       # RMS threshold for “speaking”
) -> str:
    """
    Record until you stop speaking (i.e. `silence_duration` of quiet)
    or until `timeout` seconds elapse, then transcribe via Whisper.
    """
    chunk_samples     = int(fs * chunk_ms / 1000)
    max_silent_frames = int(silence_duration * 1000 / chunk_ms)

    frames = []
    silent_count = 0
    speaking_started = False
    start_time = time.time()

    print(f"{Fore.CYAN}Listening… speak, then stay silent to end.{Style.RESET_ALL}")
    with sd.InputStream(samplerate=fs, channels=1, dtype='int16') as stream:
        while True:
            data, _ = stream.read(chunk_samples)
            frames.append(data.copy())

            amp = np.abs(data).mean()
            if amp > threshold:
                speaking_started = True
                silent_count = 0
            elif speaking_started:
                silent_count += 1

            if (speaking_started and silent_count > max_silent_frames) \
               or (time.time() - start_time > timeout):
                break

    recording = np.concatenate(frames, axis=0)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, recording, fs)
        print(f"{Fore.BLUE}Transcribing with Whisper (medium.en) on {device.upper()}…{Style.RESET_ALL}")
        result = asr.transcribe(
            tmp.name,
            fp16=(device != "cpu"),  # use fp16 on GPU/MPS
            condition_on_previous_text=False
        )
    os.remove(tmp.name)

    text = result.get("text", "").strip()
    if text:
        print(f"{Fore.YELLOW}You: {text}{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}[Error] No speech detected or too quiet.{Style.RESET_ALL}")
    return text

# ————————————————————————————————————————————————————————————————————————————————
# 1) Ollama Chat setup
client = ollama.Client()
conversation_history = [
    {
        "role": "system",
        "content": (
            "You are RECAP, an educational AI assistant. "
            "Answer concisely by default. If the user asks for details or examples, "
            "then provide a more in-depth explanation."
            "By default respond in English unless instructed otherwise."
            "Avoid speaking in bulletted points."
        )
    }
]

# ————————————————————————————————————————————————————————————————————————————————
# 2) Main RECAP loop
greeting = "Hello! I’m RECAP, your AI assistant. How can I help you today?"
print(f"{Fore.CYAN}RECAP: {greeting}{Style.RESET_ALL}")
speak(greeting)

FAREWELL_TOKENS = [
    # explicit exits
    "exit", "quit", "farewell", "goodbye"

    # complete “thank you” closers
    "thank you for your help",
    "thank you so much",
    "thanks for your help",
    "thanks for everything",
    "thanks, that's all",
    "thank you, that's all",
    "that'll be all"

    # indicate no more needs
    "that's all i needed",
    "that is all i needed",
    "i'm done",
    "we're all set",
    "all set",

    # no further inquiries
    "no more questions",
    "nothing else i need",
    "don't need anything else",

    # chat‐specific closers
    "end chat",
    "close chat",
    "stop now",

    # polite sign-offs
    "see you later",
    "take care",
    "have a great day",
    "have a fantastic day",
    "have a good day",
    "take care",
    "have a good one",
    "it was a pleasure"
]

while True:
    if has_mic:
        user_text = get_voice_input()
    else:
        user_text = input(f"{Fore.CYAN}Type your input:{Style.RESET_ALL} ")

    if not user_text:
        continue

    lower_user = user_text.lower()
    if any(tok in lower_user for tok in FAREWELL_TOKENS):
        prompt = (
            "The user is done. Respond with one concise, friendly farewell."
        )
        msgs = [conversation_history[0], {"role": "user", "content": prompt}]
        resp = client.chat(model="mistral", messages=msgs)
        farewell = resp["message"]["content"].strip()
        print(f"{Fore.MAGENTA}RECAP: {farewell}{Style.RESET_ALL}")
        speak(farewell)
        break

    conversation_history.append({"role": "user", "content": user_text})
    resp = client.chat(model="mistral", messages=conversation_history)
    bot_reply = resp["message"]["content"].strip()

    print(f"{Fore.GREEN}RECAP: {bot_reply}{Style.RESET_ALL}")
    speak(bot_reply)

    if any(tok in bot_reply.lower() for tok in FAREWELL_TOKENS):
        break
    conversation_history.append({"role": "assistant", "content": bot_reply})
