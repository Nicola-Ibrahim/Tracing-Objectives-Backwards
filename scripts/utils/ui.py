import sys

# --- ANSI Color/Style Constants ---
BOLD = "\033[1m"
DIM = "\033[2m"
BLUE = "\033[94m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

# --- Formatting Constants ---
INDENT_STEP = "  "
INDENT_OUTPUT = "      "


def log_header(text: str) -> None:
    """Prints a bold, high-contrast block header with consistent padding."""
    width = 60
    border = "━" * width
    print(f"\n{BOLD}{BLUE}{border}{RESET}")
    print(f"{BOLD}{BLUE} {text.upper()}{RESET}")
    print(f"{BOLD}{BLUE}{border}{RESET}\n")


def log_step(text: str) -> None:
    """Prints a structured action step with indentation."""
    print(f"{INDENT_STEP}{BOLD}{CYAN}➤ {text}...{RESET}")


def log_success(text: str) -> None:
    """Prints a success message with indentation."""
    print(f"{INDENT_STEP}{BOLD}{GREEN}✔ {text}{RESET}")


def log_error(text: str) -> None:
    """Prints an error message with indentation."""
    print(f"{INDENT_STEP}{BOLD}{RED}✖ ERROR: {text}{RESET}")


def log_warning(text: str) -> None:
    """Prints a warning message with indentation."""
    print(f"{INDENT_STEP}{BOLD}{YELLOW}⚠  {text}{RESET}")


def log_info(text: str) -> None:
    """Prints an informational message with dim styling and indentation."""
    print(f"{INDENT_STEP}{DIM}{CYAN}ℹ {text}{RESET}")


def confirm_action(prompt: str, skip_confirm: bool = False) -> bool:
    """
    Asks the user for a Y/n confirmation with consistent styling.
    Returns True if confirmed or if skip_confirm is True.
    """
    if skip_confirm:
        return True

    try:
        response = input(f"\n{BOLD}{YELLOW}❓ {prompt} (y/N): {RESET}").strip().lower()
        return response == "y"
    except KeyboardInterrupt:
        print("\n")
        sys.exit(0)
