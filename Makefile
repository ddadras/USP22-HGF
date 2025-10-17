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

# Run batch size experiment and save results
batch_size_experiment:
	@echo "Creating output folder: $(OUTPUT_DIR)"
	mkdir -p $(OUTPUT_DIR)/plots
	$(PYTHON) src/batch_size_experiment_plot.py --output_dir $(OUTPUT_DIR)

# Run hyperparameter tuning and save results
hyperparam_tuning:
	@echo "Creating output folder: $(OUTPUT_DIR)"
	mkdir -p $(OUTPUT_DIR)/models
	mkdir -p $(OUTPUT_DIR)/plots
	mkdir -p $(OUTPUT_DIR)/logs
	$(PYTHON) src/hyperparameter_tuning.py --output_dir $(OUTPUT_DIR)

# Run inference
inference:
	@echo "Creating output folder: $(OUTPUT_DIR)"
	mkdir -p $(OUTPUT_DIR)/predictions
	$(PYTHON) src/inference.py --output_dir $(OUTPUT_DIR)

# Clean all outputs
clean:
	@echo "Cleaning all outputs..."
	rm -rf outputs/*
	@echo "Clean complete!"