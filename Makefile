# Makefile for PyTorch Image Classification Project
# Automatically creates timestamped output folders for reproducibility

PYTHON ?= python3
PIP ?= pip
TIMESTAMP := $(shell date +"%Y%m%d_%H%M%S")
OUTPUT_DIR := results_report/$(TIMESTAMP)

# CUDA-specific installation for PyTorch
install:
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements.txt
	@echo "Dependencies installed successfully!"

# Run training and save outputs
train:
	@echo "Creating output folder: $(OUTPUT_DIR)"
	mkdir -p $(OUTPUT_DIR)/models
	mkdir -p $(OUTPUT_DIR)/plots
	mkdir -p $(OUTPUT_DIR)/logs
	@echo "Starting training..."
	$(PYTHON) src/train.py --output_dir $(OUTPUT_DIR)

# Run inference
inference:
	@echo "Creating output folder: $(OUTPUT_DIR)"
	mkdir -p $(OUTPUT_DIR)/predictions
	$(PYTHON) src/infer.py --output_dir $(OUTPUT_DIR)

# Clean all outputs
clean:
	@echo "Cleaning all outputs..."
	rm -rf outputs/*
	@echo "Clean complete!"