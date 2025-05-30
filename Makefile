SHELL := /bin/bash
VENV  := .venv
PYTHON := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip

export PATH := $(abspath $(VENV))/bin:$(PATH)
export PYTHONPATH := $(CURDIR)/src

.PHONY: all venv install run clean

all: install run

venv:
	@python3 -m venv $(VENV)
	@echo "✔ .venv ready"

install: venv
	@echo "→ Installing system packages if on Jetson…"
ifeq ($(shell uname -m), aarch64)
	sudo apt-get update
	sudo apt-get install -y \
	  python3-pip python3-venv \
	  libsndfile1 portaudio19-dev
	@echo "→ Installing NVIDIA PyTorch wheel…"
	$(PIP) install \
	  https://repo.download.nvidia.com/jetson/common \
	  -f \
	  https://repo.download.nvidia.com/jetson/torch_stable.html \
	  torch torchvision torchaudio
else
	@echo "→ Installing x86_64 PyTorch via pip…"
	$(PIP) install torch torchvision torchaudio
endif
	@echo "→ Installing Python deps…"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✔ Dependencies installed into $(VENV)"

run:
	@cd src && python model.py

clean:
	@echo "→ Removing .venv and artifacts…"
	rm -rf $(VENV) build dist __pycache__ *.spec
	@echo "✔ Cleaned up"
