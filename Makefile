SHELL := /bin/bash
VENV  := .venv
export PATH := $(abspath $(VENV))/bin:$(PATH)
export PYTHONPATH := $(CURDIR)/src

.PHONY: all venv install run clean

all: install run

venv:
	@python3 -m venv $(VENV)
	@echo "✔ .venv created"

install: venv
	@echo "→ Jetson Nano detected? Installing system packages…"
ifeq ($(shell uname -m), aarch64)
	sudo apt-get update
	# Build & Python/runtime support + audio libs + ffmpeg + pyenv prerequisites
	sudo apt-get install -y \
	  build-essential curl git libssl-dev zlib1g-dev \
	  libbz2-dev libreadline-dev libsqlite3-dev libffi-dev \
	  python3-venv python3-pip pyenv \
	  libsndfile1 portaudio19-dev ffmpeg
	@echo "→ Upgrading venv pip & wheel…"
	pip install --upgrade pip setuptools wheel
	@echo "→ Installing PyTorch for ARM64 via extra-index-url…"
	pip install \
	  --extra-index-url https://download.pytorch.org/whl/torch_stable.html \
	  torch torchvision torchaudio
else
	@echo "→ x86_64/macOS detected: installing standard PyTorch…"
	pip install torch torchvision torchaudio
endif

	@echo "→ Installing Python requirements…"
	pip install -r requirements.txt
	@echo "✔ All dependencies installed into $(VENV)"

run:
	@cd src && python model.py

clean:
	@echo "→ Removing .venv, caches, artifacts…"
	rm -rf $(VENV) build dist __pycache__ *.spec
	@echo "✔ Cleaned up"
