

# Catch all targets and store them in a variable
COMMANDS := $(shell grep -E '^[a-zA-Z0-9_-]+:' Makefile | sed 's/:.*//')

# Define color codes
RED    := \033[31m
GREEN  := \033[32m
YELLOW := \033[33m
BLUE   := \033[34m
RESET  := \033[0m


UV := uv run
UVX := uvx

.PHONY: $(COMMANDS)  # Declare all commands as PHONY

list:  # List all commands and their descriptions
	@echo "$(YELLOW)===========================$(RESET)"
	@echo "$(YELLOW)      Available Commands    $(RESET)"
	@echo "$(YELLOW)===========================$(RESET)"
	@printf "$(GREEN)%-30s$(RESET)  %s\n" "Command" "Description"
	@echo "$(YELLOW)---------------------------  --------------------$(RESET)"
	@for cmd in $(COMMANDS); do \
		desc=$$(grep "^$$cmd:" Makefile | sed 's/.*# //'); \
		printf "$(GREEN)%-30s$(RESET)  %s\n" "$$cmd" "$$desc"; \
	done
	@echo "$(YELLOW)===========================$(RESET)"

generate:  # Generate synthetic data
	@echo "Generating data..."
	$(UV) python -m src.generating.run --problem-id 5
	@echo "Data generation complete."

analyze: # Analyze data
	@echo "Analyzing data..."
	$(UV) python -m src.analyzing.run
	@echo "Data analysis complete."

run: generate analyze  # Run both generate and analyze
	@echo "Running both generate and analyze tasks..."
