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

# Default estimators for inverse (objective->decision) and forward (decision->objective) training
INVERSE_ESTIMATOR ?= mdn
FORWARD_ESTIMATOR ?= mdn

ifdef estimator
INVERSE_ESTIMATOR := $(estimator)
FORWARD_ESTIMATOR := $(estimator)
endif
ifdef inverse_estimator
INVERSE_ESTIMATOR := $(inverse_estimator)
endif
ifdef forward_estimator
FORWARD_ESTIMATOR := $(forward_estimator)
endif

# Resolve the estimator passed to CLI targets (CLI flag > direction-specific > default)
INVERSE_TARGET_ESTIMATOR = $(if $(inverse_estimator),$(inverse_estimator),$(if $(estimator),$(estimator),$(INVERSE_ESTIMATOR)))
FORWARD_TARGET_ESTIMATOR = $(if $(forward_estimator),$(forward_estimator),$(if $(estimator),$(estimator),$(FORWARD_ESTIMATOR)))

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
	$(PYTHON) -m src.modules.optimization_engine.cli.generating.generate_dataset --problem-id 5
	@echo "$(GREEN)Data generation complete.$(RESET)"

.PHONY: data-visualize
data-visualize:  # Visualize the generated data
	@echo "$(BLUE)Visualizing generated data...$(RESET)"
	$(PYTHON) -m src.modules.optimization_engine.cli.visualizing.visualize_dataset
	@echo "$(GREEN)Data visualization complete.$(RESET)"

.PHONY: model-train-inverse
model-train-inverse:  # Train an inverse model (objectives -> decisions) using a train/test split
	@echo "$(BLUE)Training a single model (standard workflow)...$(RESET)"
	$(PYTHON) -m src.modules.optimization_engine.cli.training.train_inverse_model_standard --estimation $(INVERSE_TARGET_ESTIMATOR)
	@echo "$(GREEN)Model training complete.$(RESET)"

.PHONY: model-train-inverse-cv
model-train-inverse-cv:  # Train an inverse model with k-fold cross-validation
	@echo "$(BLUE)Training a single model with cross-validation...$(RESET)"
	$(PYTHON) -m src.modules.optimization_engine.cli.training.train_inverse_model_cv --estimation $(INVERSE_TARGET_ESTIMATOR)
	@echo "$(GREEN)Cross-validation training complete.$(RESET)"

.PHONY: model-train-inverse-grid
model-train-inverse-grid:  # Run grid search + CV for an inverse model
	@echo "$(BLUE)Running grid search for a single model...$(RESET)"
	$(PYTHON) -m src.modules.optimization_engine.cli.training.train_inverse_model_grid_search --estimation $(INVERSE_TARGET_ESTIMATOR)
	@echo "$(GREEN)Grid search training complete.$(RESET)"

.PHONY: model-train-forward
model-train-forward:  # Train a forward model (decisions -> objectives) using a train/test split
	@echo "$(BLUE)Training a forward model (standard workflow)...$(RESET)"
	$(PYTHON) -m src.modules.optimization_engine.cli.training.train_forward_model standard --estimation $(FORWARD_TARGET_ESTIMATOR)
	@echo "$(GREEN)Forward model training complete.$(RESET)"


.PHONY: model-generate-decision
model-generate-decision:  # Use a trained model to generate a decision for a target objective
	@echo "$(BLUE)Generating decision from a trained model...$(RESET)"
	$(PYTHON) -m src.modules.optimization_engine.cli.generating.generate_decision --estimator $(INVERSE_TARGET_ESTIMATOR)
	@echo "$(GREEN)Decision generation complete.$(RESET)"

.PHONY: model-validate-inverse
model-validate-inverse:  # Validate an inverse model using a forward simulator
	@echo "$(BLUE)Validating inverse model...$(RESET)"
	$(PYTHON) -m src.modules.optimization_engine.cli.assurning.validate_inverse_model --estimator $(INVERSE_TARGET_ESTIMATOR)
	@echo "$(GREEN)Inverse model validation complete.$(RESET)"

.PHONY: assurance-calibrate-validation
assurance-calibrate-validation:  # Fit and persist assurance calibrators for decision validation
	@echo "$(BLUE)Calibrating decision validation gates...$(RESET)"
	$(PYTHON) -m src.modules.optimization_engine.cli.assurning.calibrate_decision_validation
	@echo "$(GREEN)Decision validation calibration complete.$(RESET)"

.PHONY: model-visualize-inverse
model-visualize-inverse:  # Visualize diagnostics for an inverse model (objectives -> decisions)
	@echo "$(BLUE)Visualizing inverse model performance...$(RESET)"
	$(PYTHON) -m src.modules.optimization_engine.cli.visualizing.visualize_model_performance --estimator $(INVERSE_TARGET_ESTIMATOR) --mapping-direction inverse $(if $(model_number),--model-number $(model_number),)
	@echo "$(GREEN)Inverse model performance visualization complete.$(RESET)"

.PHONY: model-visualize-forward
model-visualize-forward:  # Visualize diagnostics for a forward model (decisions -> objectives)
	@echo "$(BLUE)Visualizing forward model performance...$(RESET)"
	$(PYTHON) -m src.modules.optimization_engine.cli.visualizing.visualize_model_performance --estimator $(FORWARD_TARGET_ESTIMATOR) --mapping-direction forward $(if $(model_number),--model-number $(model_number),)
	@echo "$(GREEN)Forward model performance visualization complete.$(RESET)"


# ====================================================================================
# Default Targets
# ====================================================================================

# Set the default goal to 'help' if no target is specified
.DEFAULT_GOAL := help
