SHELL := /bin/bash
VENV  := .venv

export VIRTUAL_ENV := $(abspath $(VENV))
export PATH         := $(VENV)/bin:$(PATH)

.PHONY: all venv install run clean

all: install run

# 1) create the venv if it doesn't exist
venv:
	@test -d $(VENV) || python3 -m venv $(VENV)
	@echo "✔ .venv ready"

# 2) install/upgrade pip + everything in requirements.txt
install: venv
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✔ Dependencies installed into $(VENV)"

# 3) run your code: now 'python' here is from .venv/bin/python
run:
	cd src && python model.py

# 4) remove the venv and all caches/artifacts
clean:
	@echo "→ Removing .venv and build caches"
	rm -rf $(VENV) build dist __pycache__ *.spec
	@echo "✔ Cleaned up"
