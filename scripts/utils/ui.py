import logging
import sys

# --- ANSI Color/Style Constants ---
_BOLD = "\033[1m"
_DIM = "\033[2m"
_BLUE = "\033[94m"
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_RESET = "\033[0m"

# --- Formatting Constants ---
_INDENT_STEP = "  "
_INDENT_OUTPUT = "      "


class ConsoleFormatter(logging.Formatter):
    """A simple formatter that just returns the message as-is."""

    def format(self, record):
        return record.getMessage()


class Logger:
    """A wrapper class for structured logging in the development environment."""

    def __init__(self, name: str = "dev", level: int = logging.INFO):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)

        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(ConsoleFormatter())
            self._logger.addHandler(handler)

    def header(self, text: str) -> None:
        """Logs a bold, high-contrast block header."""
        width = 60
        border = "━" * width
        self._logger.info(f"\n{_BOLD}{_BLUE}{border}{_RESET}")
        self._logger.info(f"{_BOLD}{_BLUE} {text.upper()}{_RESET}")
        self._logger.info(f"{_BOLD}{_BLUE}{border}{_RESET}\n")

    def step(self, text: str) -> None:
        """Logs a structured action step."""
        self._logger.info(f"{_INDENT_STEP}{_BOLD}{_CYAN}➤ {text}...{_RESET}")

    def success(self, text: str) -> None:
        """Logs a success message."""
        self._logger.info(f"{_INDENT_STEP}{_BOLD}{_GREEN}✔ {text}{_RESET}")

    def error(self, text: str) -> None:
        """Logs an error message."""
        self._logger.error(f"{_INDENT_STEP}{_BOLD}{_RED}✖ ERROR: {text}{_RESET}")

    def warning(self, text: str) -> None:
        """Logs a warning message."""
        self._logger.warning(f"{_INDENT_STEP}{_BOLD}{_YELLOW}⚠  {text}{_RESET}")

    def info(self, text: str) -> None:
        """Logs an informational message."""
        self._logger.info(f"{_INDENT_STEP}{_DIM}{_CYAN}ℹ {text}{_RESET}")

    def output(self, text: str) -> None:
        """
        Logs dimmed, indented output for command results.
        Automatically handles multiline strings.
        """
        lines = text.strip().split("\n")
        for line in lines:
            self._logger.info(f"{_INDENT_OUTPUT}{_DIM}{line}{_RESET}")

    def error_output(self, text: str) -> None:
        """
        Logs red, indented error output for failed commands.
        Automatically handles multiline strings.
        """
        lines = text.strip().split("\n")
        for line in lines:
            self._logger.error(f"{_INDENT_OUTPUT}{_RED}{line}{_RESET}")

    def info_step(self, text: str) -> None:
        """Logs a dimmed, indented info message (e.g., 'Skipped by user')."""
        self._logger.info(f"{_INDENT_OUTPUT}{_DIM}{text}{_RESET}")

    def confirm(self, prompt: str, skip_confirm: bool = False) -> bool:
        """
        Asks the user for a Y/n confirmation.
        Returns True if confirmed or if skip_confirm is True.
        """
        if skip_confirm:
            return True

        try:
            response = (
                input(f"\n{_BOLD}{_YELLOW}❓ {prompt} (y/N): {_RESET}").strip().lower()
            )
            return response == "y"
        except KeyboardInterrupt:
            self._logger.info("\n")
            sys.exit(0)


# singleton instance
logger = Logger()
