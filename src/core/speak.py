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
    "arb": "Zeina",
    "ar-AE": "Hala",
    "nl-BE": "Lisa",
    "ca-ES": "Arlet",
    "cs-CZ": "Jitka",
    "yue-CN": "Hiujin",
    "cmn-CN": "Zhiyu",
    "da-DK": "Naja",
    "nl-NL": "Laura",
    "en-AU": "Nicole",
    "en-GB": "Amy",
    "en-IN": "Aditi",
    "en-IE": "Niamh",
    "en-NZ": "Aria",
    "en-SG": "Jasmine",
    "en-ZA": "Ayanda",
    "en-US": "Danielle",
    "en-GB-WLS": "Geraint",
    "fi-FI": "Suvi",
    "fr-FR": "Celine",
    "fr-BE": "Isabelle",
    "fr-CA": "Chantal",
    "de-DE": "Marlene",
    "de-AT": "Hannah",
    "de-CH": "Sabrina",
    "hi-IN": "Kajal",
    "is-IS": "Dora",
    "it-IT": "Carla",
    "ja-JP": "Mizuki",
    "ko-KR": "Seoyeon",
    "nb-NO": "Liv",
    "pl-PL": "Ewa",
    "pt-BR": "Camila",
    "pt-PT": "Ines",
    "ro-RO": "Carmen",
    "ru-RU": "Tatyana",
    "es-ES": "Conchita",
    "es-MX": "Mia",
    "es-US": "Lupe",
    "sv-SE": "Astrid",
    "tr-TR": "Filiz",
    "cy-GB": "Gwyneth",
}

VOICE_ENGINES: dict[str, set[str]] = {}
def _load_voice_engines():
    # This will page through all voices if necessary
    paginator = polly.get_paginator("describe_voices")
    for page in paginator.paginate():
        for v in page["Voices"]:
            VOICE_ENGINES[v["Id"]] = set(v.get("SupportedEngines", []))

_load_voice_engines()

def _select_engine(voice_id: str) -> str:
    engines = VOICE_ENGINES.get(voice_id, set())
    # prefer neural, then standard, then any other
    for choice in ("neural", "standard"):
        if choice in engines:
            return choice
    # if it has some other engine (generative/long-form), just pick that
    return next(iter(engines), "neural")

USER_VARIANT_CHOICE: dict[str, str] = {}

def _choose_variant(base: str) -> str:
    variants = [
        k for k in VOICE_MAP
        if k.split("-", 1)[0].lower() == base.lower()
    ]
    print(f"\nMultiple {base!r} voices available:")
    for i, code in enumerate(variants, start=1):
        print(f"  {i}) {code} → {VOICE_MAP[code]}")
    while True:
        sel = input(f"Pick a {base} voice [1-{len(variants)}]: ")
        if sel.isdigit() and 1 <= (idx := int(sel)) <= len(variants):
            USER_VARIANT_CHOICE[base] = variants[idx-1]
            return variants[idx-1]
        print("  Invalid choice, try again.")

def _normalize_lang(code: str) -> str:
    """
    1) If the user has previously picked a variant for this base, use that
    2) Else if code exactly matches a VOICE_MAP key, use it
    3) Else if it's a base with multiple variants, prompt (or use stored)
    4) Else fallback to en-US
    """
    code_norm = code.strip()
    base = code_norm.split("-", 1)[0]

    # 1) user override always wins
    if base in USER_VARIANT_CHOICE:
        chosen = USER_VARIANT_CHOICE[base]
        return VOICE_MAP[chosen]

    # 2) literal match
    if code_norm in VOICE_MAP:
        return VOICE_MAP[code_norm]

    # 3) bare-base logic (same as before)
    variants = [
        k for k in VOICE_MAP
        if k.split("-", 1)[0].lower() == base.lower()
    ]
    if len(variants) > 1:
        chosen = USER_VARIANT_CHOICE.get(base) or _choose_variant(base)
        return VOICE_MAP[chosen]
    if len(variants) == 1:
        return VOICE_MAP[variants[0]]

    # 4) ultimate fallback
    return VOICE_MAP["en-US"]
# ─── 4) speak() ─────────────────────────────────────────────────────────────
def speak(text: str, play: bool, language: str = "en") -> None:
    """
    Synthesize `text` via Amazon Polly and play it.
    """
    voice_id = _normalize_lang(language)
    engine = _select_engine(voice_id)
    resp = polly.synthesize_speech(
        Text=text,
        OutputFormat="pcm",
        VoiceId=voice_id,
        Engine=engine,
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
