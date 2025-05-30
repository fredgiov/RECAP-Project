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
ifeq ($(shell uname -m), aarch64)
	@echo "→ Jetson Nano detected: installing system packages…"
	sudo apt-get update
	sudo apt-get install -y \
	  build-essential curl git libssl-dev zlib1g-dev \
	  libbz2-dev libreadline-dev libsqlite3-dev libffi-dev liblzma-dev \
	  python3-venv python3-pip python3-dev python3-distutils \
	  python3-setuptools python3-wheel pkg-config \
	  libsndfile1-dev portaudio19-dev ffmpeg \
	  rustc cargo \
	  libblas-dev liblapack-dev libatlas-base-dev gfortran \
	  swig cmake default-jdk-headless
	@echo "→ Upgrading pip, setuptools, and wheel…"
	pip install --upgrade pip setuptools wheel
	@echo "→ Installing ARM64 PyTorch via extra-index-url…"
	pip install \
	  --extra-index-url https://download.pytorch.org/whl/torch_stable.html \
	  torch torchvision torchaudio
	@echo "→ Installing Python dependencies (except TTS)…"
	pip install -r requirements.txt
	@echo "→ Installing Coqui TTS with no dependencies…"
	pip install TTS --no-deps
else
	@echo "→ Non-ARM64 host: installing standard PyTorch…"
	pip install torch torchvision torchaudio
	@echo "→ Installing Python dependencies…"
	pip install -r requirements.txt
endif
	@echo "✔ All dependencies installed into $(VENV)"

run:
	@cd src && python model.py

clean:
	@echo "→ Cleaning up…"
	rm -rf $(VENV) build dist __pycache__ *.spec
	@echo "✔ Done"