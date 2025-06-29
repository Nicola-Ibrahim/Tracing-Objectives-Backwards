# Catch all targets and store them in a variable. This makes the `help` command self-documenting.
COMMANDS := $(shell grep -E '^[a-zA-Z0-9_-]+:.*' Makefile | sed 's/:.*//')

# Define standard color codes for better readability in the terminal
RED    := \033[31m
GREEN  := \033[32m
YELLOW := \033[33m
BLUE   := \033[34m
RESET  := \033[0m

# ====================================================================================
# General & Helper Commands
# ====================================================================================

# Helper command to run Python scripts within the 'uv' virtual environment.
# 'uv run' executes a script, while 'uvx' runs an arbitrary command.
UV := uv run

.PHONY: help
help:  # List all available commands and their descriptions
	@echo "$(YELLOW)================================================================$(RESET)"
	@echo "$(YELLOW)                   🚀 Project Commands 🚀                    $(RESET)"
	@echo "$(YELLOW)================================================================$(RESET)"
	@printf "$(GREEN)%-35s$(RESET)  %s\n" "Command" "Description"
	@echo "$(YELLOW)-----------------------------------  --------------------------$(RESET)"
	@for cmd in $(COMMANDS); do \
		desc=$$(grep "^$$cmd:" Makefile | sed 's/.*# //'); \
		printf "$(GREEN)%-35s$(RESET)  %s\n" "$$cmd" "$$desc"; \
	done
	@echo "$(YELLOW)================================================================$(RESET)"

.PHONY: install
install:  # Install project dependencies using the uv lock file
	@echo "$(BLUE)Installing project dependencies...$(RESET)"
	uv pip install -r requirements.txt
	@echo "$(GREEN)Dependencies installed successfully!$(RESET)"

# ====================================================================================
# Data Pipeline Commands
# ====================================================================================

.PHONY: data-generate
data-generate:  # Generate synthetic Pareto front data for a specified problem
	@echo "$(BLUE)Generating synthetic data...$(RESET)"
	$(UV) python -m src.modules.optimization_engine.cli.make_pareto --problem-id 5
	@echo "$(GREEN)Data generation complete.$(RESET)"

.PHONY: data-analyze
data-analyze:  # Analyze and visualize the generated synthetic data
	@echo "$(BLUE)Analyzing generated data...$(RESET)"
	$(UV) python -m src.modules.optimization_engine.cli.analyze_data
	@echo "$(GREEN)Data analysis complete.$(RESET)"

.PHONY: data-process
data-process: data-generate data-analyze  # Run the full data generation and analysis pipeline
	@echo "$(GREEN)✔️ Full data processing pipeline completed successfully.$(RESET)"

# ====================================================================================
# Model Training Commands
# ====================================================================================

.PHONY: train-interpolator
train-interpolator:  # Train an inverse decision mapper using the synthetic data
	@echo "$(BLUE)Training interpolator model...$(RESET)"
	$(UV) python -m src.modules.optimization_engine.cli.train_interpolator
	@echo "$(GREEN)Interpolator training complete.$(RESET)"

# ====================================================================================
# Model Training Commands
# ====================================================================================

.PHONY: analyze-performance
analyze-performance:  # Analyze and visualize the performance metrics of trained models
	@echo "$(BLUE)Analyzing model performance...$(RESET)"
	$(UV) python -m  src.modules.optimization_engine.cli.analyze_performance
	@echo "$(GREEN)Performance analysis complete.$(RESET)"

# ====================================================================================
# Maintenance Commands
# ====================================================================================

.PHONY: clean
clean:  # Clean up generated data, cache files, and trained models
	@echo "$(RED)Cleaning up generated files...$(RESET)"
	rm -rf src/data/raw/synthetic/*
	rm -rf models/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -r {} +
	@echo "$(GREEN)Cleanup complete.$(RESET)"

# ====================================================================================
# Default & All Targets
# ====================================================================================

.PHONY: all
all: data-process train-interpolator  # Run the entire data processing and training workflow

# Declare all targets as PHONY to prevent conflicts with files of the same name
.PHONY: $(COMMANDS)

# Set the default goal to 'help' if no target is specified
.DEFAULT_GOAL := help