# Makefile

PYTHON := python
ENV_NAME := imgcls

# Default target
help:
	@echo "Usage:"
	@echo "  make setup     - create virtualenv & install deps"
	@echo "  make train     - run training script"
	@echo "  make clean     - remove caches and results"
	@echo "  make freeze    - export current deps to requirements.txt"

setup:
	python -m venv .venv
	. .venv/bin/activate; pip install --upgrade pip
	. .venv/bin/activate; pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124

train:
	. .venv/bin/activate; $(PYTHON) src/train.py

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache results_report/*.png results_report/*.csv results_report/*.pdf

freeze:
	. .venv/bin/activate; pip freeze > requirements.txt
