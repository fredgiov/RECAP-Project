import os
import time
import tempfile
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
# Prerequisite: Ensure ffmpeg is installed on your Mac:
#    brew install ffmpeg
#
# 1) ASR: Load Whisper medium.en on CPU to densify sparse weights, then move to MPS
if not torch.backends.mps.is_available():
    print(f"{Fore.RED}[Error] MPS backend not available—cannot continue.{Style.RESET_ALL}")
    exit(1)

print(f"{Fore.CYAN}Loading Whisper medium.en on CPU to patch sparse weights…{Style.RESET_ALL}")
asr = whisper.load_model("medium.en", device="cpu")

# Densify any sparse buffers so MPS inference won’t fail
for name, buf in list(asr.named_buffers()):
    if buf.layout == torch.sparse_coo:
        dense = buf.to_dense()
        asr.register_buffer(name, dense)

print(f"{Fore.CYAN}Moving Whisper model to MPS…{Style.RESET_ALL}")
asr = asr.to("mps")

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
    chunk_samples    = int(fs * chunk_ms / 1000)
    max_silent_chunks = int(silence_duration * 1000 / chunk_ms)

    frames = []
    silent_chunks = 0
    speaking_started = False
    start_time = time.time()

    print(f"{Fore.CYAN}Listening… speak, then stay silent to end.{Style.RESET_ALL}")
    with sd.InputStream(samplerate=fs, channels=1, dtype='int16') as stream:
        while True:
            data, _ = stream.read(chunk_samples)
            frames.append(data.copy())

            # simple RMS-style amplitude check
            amp = np.abs(data).mean()
            if amp > threshold:
                speaking_started = True
                silent_chunks = 0
            elif speaking_started:
                silent_chunks += 1

            # stop if we've been quiet long enough, or hit the overall timeout
            if (speaking_started and silent_chunks > max_silent_chunks) \
               or (time.time() - start_time > timeout):
                break

    # stitch all the chunks together
    recording = np.concatenate(frames, axis=0)

    # write to temp WAV & run Whisper exactly as before
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, recording, fs)
        print(f"{Fore.BLUE}Transcribing with Whisper (medium.en) on MPS…{Style.RESET_ALL}")
        result = asr.transcribe(
            tmp.name,
            fp16=False,
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
# 2) Ollama Chat setup with concise‐by‐default system prompt

# Make sure you've run `ollama serve` in another terminal
client = ollama.Client()

conversation_history = [
    {
        "role": "system",
        "content": (
            "You are RECAP, an educational AI assistant. "
            "Answer concisely by default. If the user asks for details or examples, "
            "then provide a more in-depth explanation. "
            "By default you are to respond in English, unless instructed otherwise."
        )
    }
]

# ————————————————————————————————————————————————————————————————————————————————
# 3) Main RECAP loop

greeting = "Hello! I’m RECAP, your AI assistant. How can I help you today?"
print(f"{Fore.CYAN}RECAP: {greeting}{Style.RESET_ALL}")
speak(greeting)

FAREWELL_TOKENS = [
    # explicit exits
    "exit", "quit", "farewell",

    # complete “thank you” closers
    "thank you for your help",
    "thank you so much",
    "thanks for your help",
    "thanks for everything",
    "thanks, that's all",
    "thank you, that's all",

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
    "it was a pleasure"
]

while True:
    user_text = get_voice_input()
    if not user_text:
        continue

    lower_user = user_text.lower().replace("'", "")

    # if the user *says* any farewell token anywhere, exit
    if any(token in lower_user for token in FAREWELL_TOKENS):
        # Ask the model as a user to craft a single, concise farewell
        farewell_prompt = (
            "The user has indicated they are done. "
            "Please respond with a single, concise, friendly farewell and nothing else."
        )
        messages_for_farewell = [
            conversation_history[0],  # original system prompt
            {"role": "user", "content": farewell_prompt}
        ]
        response = client.chat(
            model="mistral",
            messages=messages_for_farewell,
        )
        farewell = response["message"]["content"].strip()

        print(f"{Fore.MAGENTA}RECAP: {farewell}{Style.RESET_ALL}")
        speak(farewell)
        break

    conversation_history.append({"role": "user", "content": user_text})
    response = client.chat(
        model="mistral",
        messages=conversation_history,
    )
    bot_reply = response["message"]["content"].strip()

    print(f"{Fore.GREEN}RECAP: {bot_reply}{Style.RESET_ALL}")
    speak(bot_reply)

    # NEW: if the model’s reply itself is a farewell, stop
    lower_reply = bot_reply.lower()
    if any(token in lower_reply for token in FAREWELL_TOKENS):
        break

    conversation_history.append({"role": "assistant", "content": bot_reply})
