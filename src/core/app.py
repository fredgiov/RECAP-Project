import os
import tempfile
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import torch
import json
from ollama import AsyncClient
from src.core.model import (
    ensureMac,
    determine_device,
    load_class_material,
    build_system_message,
    load_farewells,
    warmup as model_warmup,
)
from src.core.speak import speak
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Async lifespan handler replaces deprecated on_event startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr, device, client, system_message, conversation_history, FAREWELL_TOKENS
    # Startup tasks
    ensureMac()
    asr, device = determine_device()
    class_material = load_class_material()
    system_message = build_system_message(class_material)
    conversation_history = [system_message]
    FAREWELL_TOKENS = load_farewells().splitlines()
    client = AsyncClient()
    # Warm up models and TTS in background
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, model_warmup)
    yield
    # (Optional) Shutdown tasks here

# Pass lifespan to FastAPI
app = FastAPI(lifespan=lifespan)
# Serve static files
app.mount(
    "/static",
    StaticFiles(directory="src/webservice/static"),
    name="static",
)

# Globals !!!
asr = None
device = None
client: AsyncClient
system_message = None
conversation_history = []
FAREWELL_TOKENS = []

@app.get("/")
async def get_index():
    return FileResponse("src/webservice/static/index.html")

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive either a text frame or binary frame
            msg = await websocket.receive()

            # 1) Text input path
            if "text" in msg:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "text":
                    user_text = data["content"]
                    user_lang = "en"
                else:
                    # unknown text payload
                    continue

            # 2) Voice input path
            elif "bytes" in msg:
                blob = msg["bytes"]
                # write to temp file for Whisper
                with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
                    tmp.write(blob)
                    tmp_path = tmp.name
                result = asr.transcribe(
                    tmp_path,
                    fp16=(device != "cpu"),
                    condition_on_previous_text=False,
                )
                os.remove(tmp_path)
                user_text = result.get("text", "").strip()
                user_lang = result.get("language", "en")
            else:
                # ignore ping/pong or other control frames
                continue

            # Append user to history
            conversation_history.append({"role": "user", "content": user_text})

            # Farewell check
            if any(tok.lower() in user_text.lower() for tok in FAREWELL_TOKENS):
                resp = await client.chat(
                    model="gemma3:4b",
                    messages=[
                        conversation_history[0],
                        {"role":"user","content":"The user is done. Respond with a concise farewell."}
                    ]
                )
                farewell = resp["message"]["content"].strip()
                await websocket.send_text(farewell)
                speak(farewell, True, language=user_lang)
                break

            # Build message list (handle non-English)
            if user_lang == "en":
                msgs = conversation_history
            else:
                msgs = [conversation_history[0], {
                    "role":"system",
                    "content":f"Please respond in {user_lang}."
                }] + conversation_history[1:]

            # Stream the assistant’s reply
            reply_accum = ""
            async for chunk in client.chat(model="gemma3:4b", messages=msgs, stream=True):
                token = chunk.get("message", {}).get("content", "")
                if token:
                    reply_accum += token
                    await websocket.send_text(token)

            # Play full TTS and save to history
            speak(reply_accum, True, language=user_lang)
            conversation_history.append({"role":"assistant", "content": reply_accum})

    except WebSocketDisconnect:
        print("RECAP client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)