SHELL := /bin/bash
VENV := .venv
OLLAMA_MODEL ?= gemma3:4b

# Ensure project venv bin is first on PATH for all make recipes
export PATH := $(abspath $(VENV))/bin:$(PATH)
export PYTHONPATH := $(CURDIR)/src

.PHONY: all setup venv install update-settings ollama-pull ollama-serve run shell clean

all: setup ollama-pull

setup: install update-settings

venv:
	@python3 -m venv $(VENV)
	@echo "✔ .venv created"

install: venv
	@echo "→ Detected macOS: installing Homebrew packages…"
	@brew update || true
	@brew install python3 portaudio ffmpeg ollama
	@echo "→ Installing Python packages…"
	@. $(VENV)/bin/activate && pip install --upgrade pip setuptools wheel
	@. $(VENV)/bin/activate && pip install -r requirements.txt openai-whisper sounddevice soundfile ollama
	@echo "✔ Dependencies installed into $(VENV). You need to initialize aws-polly by using aws configure."

update-settings:
	@mkdir -p .vscode
	@echo "{" > .vscode/settings.json
	@echo "  \"python.defaultInterpreterPath\": \"$(CURDIR)/.venv/bin/python\"," >> .vscode/settings.json
	@echo "  \"python.terminal.activateEnvironment\": true," >> .vscode/settings.json
	@echo "  \"python.analysis.extraPaths\": [\"$(CURDIR)/src\"]" >> .vscode/settings.json
	@echo "}" >> .vscode/settings.json
	@echo "✔ VS Code settings created"

ollama-pull:
	@echo "→ Pulling Ollama model $(OLLAMA_MODEL)…"
	@ollama pull $(OLLAMA_MODEL)
	@echo "✔ Ollama model ready"

ollama-serve:
	@echo "→ Starting Ollama server (keep this terminal open)…"
	@ollama serve

run:
	@echo "→ Running model.py…"
	@python src/core/model.py

shell:
	@echo "→ Entering project shell with .venv activated (run 'exit' to leave)…"
	@. $(VENV)/bin/activate && exec $$SHELL

clean:
	@echo "→ Cleaning workspace…"
	@rm -rf $(VENV) .vscode build dist __pycache__ *.spec
	@echo "✔ Clean complete"
