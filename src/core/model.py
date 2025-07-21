# Imports --------------------------------
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
import threading
from colorama import Fore, Style, init
from src.core.speak import speak
from pynput import keyboard
from pynput.keyboard import Listener
from typing import Tuple, Any, Dict

# Initialize colorama
init(autoreset = True)

# Ensure macOS function ------------------
def ensureMac() -> None:
    if platform.system() != "Darwin":
        raise OSError("This program is macOS-specific; please run on a Mac.")

# Determine device function --------------
def determine_device() -> Tuple[Any, str]:
    if torch.backends.mps.is_available():
        device = "mps"
    else: 
        device = "cpu"
  
    print(f"{Fore.CYAN}Loading Whisper small on CPU to patch sparse weights...{Style.RESET_ALL}")
    asr = whisper.load_model("small", device = "cpu")
    # Patching sparse buffers now
    for name, buf in list(asr.named_buffers()):
        if buf.layout == torch.sparse_coo:
            asr.register_buffer(name, buf.to_dense())
    # Finalizing
    print(f"{Fore.CYAN}Moving Whisper model to {device.upper()}...{Style.RESET_ALL}")
    asr = asr.to(device)
    return asr, device

# Detect microphone function -------------
def determineIf_mic_available() -> bool:
    try:
        input_devices = [device for device in sd.query_devices() if device.get("max_input_channels", 0) > 0]
        has_mic = len(input_devices) > 0
    except Exception:
        has_mic = False
    if not has_mic:
        print(f"{Fore.YELLOW}[Warning] No microphone detected. Voice mode disabled.{Style.RESET_ALL}")
    return has_mic

# Key Toggler Setup Function -------------
def setup_hotkeys_and_listeners() -> Listener:
    global use_tts, use_voice

    # Toggle modes
    def toggle_mode():
        global use_voice
        use_voice = not use_voice
        mode = "VOICE" if use_voice else "TEXT"
        print(f"\n{Fore.MAGENTA}*** Switched to {mode} mode ***{Style.RESET_ALL}")

    def toggle_tts():
        global use_tts
        use_tts = not use_tts
        state = "ON" if use_tts else "OFF"
        print(f"\n{Fore.MAGENTA}*** Voice output turned {state} ***{Style.RESET_ALL}")

    def stop_speaking():
        sd.stop()
        print(f"\n{Fore.MAGENTA}*** Current voice output stopped ***{Style.RESET_ALL}")

    hotkey_mode = keyboard.HotKey(
        keyboard.HotKey.parse('<cmd>+/'),
        toggle_mode
    )

    hotkey_tts = keyboard.HotKey(
        keyboard.HotKey.parse('<cmd>+\\'),
        toggle_tts
    )  

    hotkey_stopSpeaking = keyboard.HotKey(
        keyboard.HotKey.parse('<cmd>+d'),
        stop_speaking
    )

    def on_press(key):
        hotkey_mode.press(key)
        hotkey_tts.press(key)
        hotkey_stopSpeaking.press(key)

    def on_release(key):
        hotkey_mode.release(key)
        hotkey_tts.release(key)
        hotkey_stopSpeaking.release(key)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()

    print(
        f"{Fore.CYAN}Press Cmd+/ at any time to toggle between voice/text input, Cmd+\ to toggle voice output and Cmd+d to turn off current voice output.{Style.RESET_ALL}"
    )

    return listener

# Get voice input function ---------------
def get_voice_input(
    timeout: float = 255.0,
    silence_duration: float = 2.5,
    fs: int = 16000,
    chunk_ms: int = 30,
    threshold: float = 500.0,
) -> tuple[str, str]:
    """
    Record until silence or timeout, then transcribe via Whisper.
    Returns (text, language_code).
    """
    chunk_samples = int(fs * chunk_ms / 1000)
    max_silent_frames = int(silence_duration * 1000 / chunk_ms)

    frames = []
    silent_count = 0
    speaking_started = False
    start_time = time.time()

    print(f"{Fore.CYAN}Listening... speak, then stay silent to end.{Style.RESET_ALL}")
    with sd.InputStream(samplerate = fs, channels = 1, dtype = "int16") as stream:
        while True:
            data, _ = stream.read(chunk_samples)
            frames.append(data.copy())
            amp = np.abs(data).mean()
            if amp > threshold:
                speaking_started = True
                silent_count = 0
            elif speaking_started:
                silent_count += 1
            if (speaking_started and silent_count >= max_silent_frames) or (time.time() - start_time > timeout):
                break
  
    recording = np.concatenate(frames, axis = 0)
    with tempfile.NamedTemporaryFile(suffix = ".wav", delete = False) as tmp:
        sf.write(tmp.name, recording, fs)
        print(f"{Fore.BLUE}Transcribing with Whisper (small) on {device.upper()}...{Style.RESET_ALL}")
        result = asr.transcribe(
            tmp.name,
            fp16 = (device != "cpu"),
            condition_on_previous_text = False
        )
    os.remove(tmp.name)

    text = result.get("text", "").strip()
    lang = result.get("language", "en")
    if text:
        print(f"{Fore.YELLOW}You ({lang}): {text}{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}[Error] No speech detected or too quiet.{Style.RESET_ALL}")
    return text, lang

# Load system content function -----------
def load_system_content() -> str:
    base = os.path.dirname(__file__)
    txt_path = os.path.join(base, "OLD_SYSTEM_CONTENT.txt")
    if not os.path.isfile(txt_path):
        raise RuntimeError(f"Cannot find {txt_path}. Make sure the filename or location is correct.")
    with open(txt_path, "r", encoding = "utf-8") as f:
        return f.read().rstrip()

# Load class material function -----------
def load_class_material() -> str:
    base = os.path.dirname(__file__)
    txt_path = os.path.join(base, "CPEG484Mock.txt")
    if not os.path.isfile(txt_path):
        raise RuntimeError(f"Cannot find {txt_path}. Make sure the filename or location is correct.")
    with open(txt_path, "r", encoding = "utf-8") as f:
        return f.read().rstrip()

# Build system message function ----------
def build_system_message(class_material: str) -> Dict[str, str]:
    old = load_system_content()
    boundary = "-----CLASS MATERIAL"
    content = "\n\n".join([
        old,
        "THIS IS YOUR CLASS MATERIAL (DO NOT INVENT BEYOND IT):",
        f"{boundary} STARTS HERE -----",
        class_material,
        f"{boundary} ENDS HERE -----",
        (
        "Whenever you answer, your explanation MUST be rooted in the above class material."
        "Do not hallucinate or introduce facts not present in that text."
        "If the user requests examples or deeper details, pull them strictly from this content."
        "Respond in English by default; avoid bullet points unless absolutely necessary."
        )
    ])
    return {"role": "system", "content": content}

# Definition of Farewell Tokens ----------
def load_farewells() -> str:
    base = os.path.dirname(__file__)
    txt_path = os.path.join(base, "FAREWELL_TOKENS.txt")
    if not os.path.isfile(txt_path):
        raise RuntimeError(f"Cannot find {txt_path}. Make sure the filename or location is correct.")
    with open(txt_path, "r", encoding = "utf-8") as f:
        FAREWELL_TOKENS = f.read()
    return FAREWELL_TOKENS

# Warm-up system function ----------------
def warmup() -> None:
    global model_ready
    try:
        # Model Specific Warmup
        ollama.chat(model="gemma3:4b", messages=[{"role": "user", "content": "Just answer: Who designed you?"}])
        model_ready = True
        print(f"{Fore.GREEN}Model warmed up and ready for conversation.{Style.RESET_ALL}")

        # TTS specific warmup
        speak("TTS warmup", False, "en")
    except Exception as e:
        raise RuntimeError(f"{Fore.RED}Model could not be warmed up. {e}.{Style.RESET_ALL}")

# Greet user function --------------------
def greet() -> None:
    greeting = "Hello! I'm RECAP, your AI assistant. How can I help you today?"
    print(f"{Fore.CYAN}RECAP: {greeting}{Style.RESET_ALL}")
    if use_tts:
        speak(greeting, True, language = "en")

# Chat with user function ----------------
def chat(modelIn: str) -> None:
    try:
        while True:
            if use_voice and has_mic:
                try:
                    user_text, user_lang = get_voice_input()
                except Exception:
                    print(f"{Fore.YELLOW}[Warning] Voice failed, switching to text input for this turn.{Style.RESET_ALL}")
                    user_text = input(f"{Fore.CYAN}[TEXT] > {Style.RESET_ALL}")
                    user_lang = "en"
            else:
                user_text = input(f"{Fore.CYAN}[TEXT] > {Style.RESET_ALL}")
                user_lang = "en"

            if not user_text:
                continue

            lower_user = user_text.lower()
            if any(tok in lower_user for tok in FAREWELL_TOKENS):
                conversation_history.append({"role": "user", "content": user_text})
                resp = client.chat(
                    model = modelIn,
                    messages = [conversation_history[0], {"role": "user", "content": "The user is done. Respond with one concise, friendly farewell."}]
                )
                farewell = resp["message"]["content"].strip()
                print(f"{Fore.MAGENTA}RECAP: {farewell}{Style.RESET_ALL}")
                if use_tts:
                    speak(farewell, True, language=user_lang)
                break

            conversation_history.append({"role": "user", "content": user_text})
            msgs = ([conversation_history[0]] + conversation_history[1:]) if user_lang == "en" else ([conversation_history[0], {"role": "system", "content": f"Please respond in {user_lang}."}] + conversation_history[1:])
            resp = client.chat(model = modelIn, messages = msgs)
            bot_reply = resp["message"]["content"].strip()

            print(f"{Fore.GREEN}RECAP: {bot_reply}{Style.RESET_ALL}")
            if use_tts:
                speak(bot_reply, True, language = user_lang)
            conversation_history.append({"role": "assistant", "content": bot_reply})
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Interrupted! Exiting...{Style.RESET_ALL}")

# Calls ----------------------------------
if __name__ == "__main__":
    # 1) Platform and Devices
    ensureMac()
    asr, device = determine_device()
    has_mic = determineIf_mic_available()
    use_voice = has_mic
    use_tts = True

    # 2a) Hotkeys
    setup_hotkeys_and_listeners()
    
    # 2b) Warmup
    warmup_thread = threading.Thread(target = warmup)
    warmup_thread.start()
    warmup_thread.join()

    # 3) Load all texts
    FAREWELL_TOKENS = load_farewells().splitlines()
    class_material = load_class_material()
    system_message = build_system_message(class_material)
    conversation_history = [system_message]

    # 4) Initialize chat client
    client = ollama.Client()

    # 5) Boot and talk
    greet()
    chat("gemma3:4b")
