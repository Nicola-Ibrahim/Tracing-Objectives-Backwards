import logging
from typing import Any, Dict, Optional

from ...domain.common.interfaces.base_logger import BaseLogger


# Define ANSI color codes
class LogColors:
    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors (less common for logs, but available)
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


class ColoredFormatter(logging.Formatter):
    """
    A custom formatter that adds color to log messages based on their level.
    """

    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    LOG_COLORS = {
        logging.DEBUG: LogColors.BRIGHT_BLACK,  # Or LogColors.CYAN, this is already defined
        logging.INFO: LogColors.GREEN,
        logging.WARNING: LogColors.YELLOW,
        logging.ERROR: LogColors.RED,
        logging.CRITICAL: LogColors.BG_RED
        + LogColors.WHITE,  # White text on red background
    }

    def format(self, record):
        log_fmt = self.FORMAT
        color = self.LOG_COLORS.get(record.levelno)
        if color:
            # Apply color to the entire message part, including levelname and message
            log_fmt = color + log_fmt + LogColors.RESET

        # Use the default formatter's functionality but with our custom format string
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class CMDLogger(BaseLogger):
    def __init__(self, name: str = "CMDLogger", level: int = logging.INFO):
        """
        Initialize the CMD logger with colorful output.

        Args:
            name (str): The name of the logger.
            level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
        """
        self.logger = logging.getLogger(name)
        # Set the logger's level based on the constructor argument
        self.logger.setLevel(level)

        # Prevent adding duplicate handlers if the logger is initialized multiple times
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = ColoredFormatter()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.propagate = (
                False  # Prevent messages from being duplicated by root logger
            )

    def log_info(self, message: str) -> None:
        """Log an informational message."""
        self.logger.info(message)

    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)

    def log_error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)

    def log_debug(self, message: str) -> None:
        """Log a debug message."""
        self.logger.debug(message)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log evaluation or training metrics."""
        if step is not None:
            self.logger.info(f"Step {step}: Metrics: {metrics}")
        else:
            self.logger.info(f"Metrics: {metrics}")

    def log_model(
        self,
        model: Any,
        name: str,
        estimator_type: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        notes: Optional[str] = None,
        collection_name: Optional[str] = None,
        step: Optional[int] = None,
    ):
        """Log model details for CMD output. Ignores some W&B specific metadata."""
        log_message = f"Logging Model: '{name}'"
        if estimator_type:
            log_message += f", Type: {estimator_type}"
        if step is not None:
            log_message += f", Step: {step}"
        if description:
            log_message += f", Desc: '{description}'"
        if parameters:
            log_message += f", Params: {parameters}"
        if metrics:
            log_message += f", Metrics: {metrics}"
        if notes:
            log_message += f", Notes: '{notes}'"
        if collection_name:
            log_message += f", Collection: '{collection_name}'"

        self.logger.info(log_message)
