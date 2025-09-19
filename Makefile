# ====================================================================================
# General & Configuration
# ====================================================================================

# Catch all targets and store them in a variable. This makes the `help` command self-documenting.
COMMANDS := $(shell grep -E '^[a-zA-Z0-9_-]+:.*' Makefile | sed 's/:.*//')

# Define standard color codes for better readability in the terminal
RED    := \033[31m
GREEN  := \033[32m
YELLOW := \033[33m
BLUE   := \033[34m
RESET  := \033[0m

# Helper command to run Python scripts within the 'uv' virtual environment.
# 'uv run' executes a script, while 'uvx' runs an arbitrary command.
PYTHON := uv run python

# ====================================================================================
# Project Management
# ====================================================================================

.PHONY: help
help:  # Display this help menu with all available commands
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
install:  # Install all project dependencies from the lock file
	@echo "$(BLUE)Installing project dependencies...$(RESET)"
	uv pip install -r requirements.txt
	@echo "$(GREEN)Dependencies installed successfully!$(RESET)"

.PHONY: clean
clean:  # Remove all generated data, trained models, and cache files
	@echo "$(RED)Cleaning up generated files...$(RESET)"
	rm -rf src/data/raw/synthetic/
	rm -rf models/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -r {} +
	@echo "$(GREEN)Cleanup complete.$(RESET)"

# ====================================================================================
# Data & Training Pipeline
# ====================================================================================

.PHONY: data-generate
data-generate:  # Generate synthetic Pareto front data for a specified problem
	@echo "$(BLUE)Generating synthetic data...$(RESET)"
	$(PYTHON) -m src.modules.optimization_engine.cli.datasets.generate_dataset --problem-id 5
	@echo "$(GREEN)Data generation complete.$(RESET)"

.PHONY: data-process
data-process:  # Generate synthetic Pareto front data for a specified problem
	@echo "$(BLUE)Processing synthetic data...$(RESET)"
	$(PYTHON) -m src.modules.optimization_engine.cli.datasets.process_dataset
	@echo "$(GREEN)Data processing complete.$(RESET)"

.PHONY: data-visualize
data-visualize:  # Visualize the generated data
	@echo "$(BLUE)Visualizing generated data...$(RESET)"
	$(PYTHON) -m src.modules.optimization_engine.cli.visualization.visualize_dataset
	@echo "$(GREEN)Data visualization complete.$(RESET)"

.PHONY: model-train-single
model-train-single:  # Train a single inverse decision mapper using a train/test split
	@echo "$(BLUE)Training a single model (standard workflow)...$(RESET)"
	$(PYTHON) -m src.modules.optimization_engine.cli.modeling.train_single_model standard
	@echo "$(GREEN)Model training complete.$(RESET)"

.PHONY: model-train-cv
model-train-cv:  # Train a single model using k-fold cross-validation
	@echo "$(BLUE)Training a single model with cross-validation...$(RESET)"
	$(PYTHON) -m src.modules.optimization_engine.cli.modeling.train_single_model cv --cv-splits 5 --estimator rbf
	@echo "$(GREEN)Cross-validation training complete.$(RESET)"

.PHONY: model-train-grid
model-train-grid:  # Run grid search with cross-validation for a single model
	@echo "$(BLUE)Running grid search for a single model...$(RESET)"
	$(PYTHON) -m src.modules.optimization_engine.cli.modeling.train_single_model grid --cv-splits 5 --estimator rbf --tune-param-name n_neighbors --tune-param-value 5 --tune-param-value 10 --tune-param-value 20 --tune-param-value 40
	@echo "$(GREEN)Grid search training complete.$(RESET)"

.PHONY: model-train-all
model-train-all:  # Train all inverse decision mappers defined in configuration
	@echo "$(BLUE)Training all configured model models...$(RESET)"
	$(PYTHON) -m src.modules.optimization_engine.cli.modeling.train_all_models
	@echo "$(GREEN)All models trained successfully.$(RESET)"

.PHONY: model-generate-decision
model-generate-decision:  # Use a trained model to generate a decision for a target objective
	@echo "$(BLUE)Generating decision from a trained model...$(RESET)"
	$(PYTHON) -m src.modules.optimization_engine.cli.modeling.generate_decision
	@echo "$(GREEN)Decision generation complete.$(RESET)"

.PHONY: model-visualize-performance
model-visualize-performance:  # Visualize the performance of trained models
	@echo "$(BLUE)Analyzing and visualizing model performance...$(RESET)"
	$(PYTHON) -m src.modules.optimization_engine.cli.visualization.visualize_model_performance
	@echo "$(GREEN)Performance analysis complete.$(RESET)"

# ====================================================================================
# Default Targets
# ====================================================================================

.PHONY: all
all: data-process model-train-all # Run the complete data and training workflow

# Set the default goal to 'help' if no target is specified
.DEFAULT_GOAL := help

# Declare all targets as PHONY to prevent conflicts with files of the same name
.PHONY: $(COMMANDS)
