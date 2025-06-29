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

print(f"{Fore.CYAN}Loading Whisper small on CPU to patch sparse weights…{Style.RESET_ALL}")
asr = whisper.load_model("small", device="cpu")

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
    input_devices = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
    has_mic = len(input_devices) > 0
except Exception:
    has_mic = False

if not has_mic:
    print(f"{Fore.YELLOW}[Warning] No microphone detected. Falling back to text input.{Style.RESET_ALL}")

txt_path = os.path.join(
    os.path.dirname(__file__),
    "CPEG484Mock.txt"
)
try:
    with open(txt_path, "r", encoding="utf-8") as f:
        class_material = f.read()
except FileNotFoundError:
    raise RuntimeError(f"Cannot find {txt_path}. Make sure the filename is correct.")

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
    with sd.InputStream(samplerate=fs, channels=1, dtype="int16") as stream:
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
        print(f"{Fore.BLUE}Transcribing with Whisper (small) on {device.upper()}…{Style.RESET_ALL}")
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

old_system_content = (
    "You are RECAP, an educational AI assistant. "
    "You were developed by the Academic Technology Services (ATS) team at the University of Delaware (UD) to work in conjunction with, "
    "other UD services such as StudyAIDE and UDcapture in order to help students with their learning, studying, "
    "and understanding of their classes. "
    "UDcapture is a service that teachers can enroll in in order to record and post lectures that they teach for, "
    "their students to access in case of not being able to attend class or for extra review. "
    "StudyAIDE is a service that teachers and students will be able to use to train you on their material, which, "
    "in turn, you will be able to access and return study options to students as well as recaps on previous lecture, "
    "material studied in class. "
    "Answer concisely by default. If the user asks for details or examples, then provide a more in-depth explanation. "
    "By default respond in English unless instructed otherwise. Avoid speaking in bulleted points. "
    "Your goal isn't to return any of these points verbatim, but as a general review of what was given."
)

system_message = {
    "role": "system",
    "content": (
        f"{old_system_content}\n\n"
        f"THIS IS YOUR CLASS MATERIAL (DO NOT INVENT BEYOND IT):\n\n"
        f"----- CLASS MATERIAL STARTS HERE -----\n\n"
        f"{class_material}\n\n"
        f"----- CLASS MATERIAL ENDS HERE -----\n\n"
        f"Whenever you answer, your explanation MUST be rooted in the above class material. "
        f"Do not hallucinate or introduce facts not present in that text. "
        f"If the user requests examples or deeper details, pull them strictly from this content. "
        f"Respond in English by default; avoid bullet points unless absolutely necessary."
    )
}

conversation_history = [system_message]

# 2) Main RECAP loop
greeting = "Hello! I’m RECAP, your AI assistant. How can I help you today?"
print(f"{Fore.CYAN}RECAP: {greeting}{Style.RESET_ALL}")
speak(greeting)

FAREWELL_TOKENS = [
    # explicit exits
    "exit", "quit", "farewell", "goodbye",

    # complete “thank you” closers
    "thank you for your help",
    "thank you so much",
    "thanks for your help",
    "thanks for everything",
    "thanks, that's all",
    "thank you, that's all",
    "that'll be all",

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
    "it was a pleasure",
    "i appreciate your help",
    "have a good rest of your day"
]

try:
    while True:
        if has_mic:
            user_text = get_voice_input()
        else:
            user_text = input(f"{Fore.CYAN}Type your input:{Style.RESET_ALL} ")

        if not user_text:
            continue

        lower_user = user_text.lower()
        if any(tok in lower_user for tok in FAREWELL_TOKENS):
            # Final farewell (synchronous so we can then exit cleanly)
            conversation_history.append({"role": "user", "content": user_text})
            prompt = "The user is done. Respond with one concise, friendly farewell."
            msgs = [conversation_history[0], {"role": "user", "content": prompt}]
            resp = client.chat(model="llama3.2:3b", messages=msgs)
            farewell = resp["message"]["content"].strip()
            print(f"{Fore.MAGENTA}RECAP: {farewell}{Style.RESET_ALL}")
            speak(farewell)
            break

        # Synchronous: append user → chat → speak → append assistant
        conversation_history.append({"role": "user", "content": user_text})
        resp = client.chat(model="llama3.2:3b", messages=conversation_history)
        bot_reply = resp["message"]["content"].strip()

        print(f"{Fore.GREEN}RECAP: {bot_reply}{Style.RESET_ALL}")
        speak(bot_reply)
        conversation_history.append({"role": "assistant", "content": bot_reply})

except KeyboardInterrupt:
    print(f"\n{Fore.RED}Interrupted! Exiting...{Style.RESET_ALL}")
