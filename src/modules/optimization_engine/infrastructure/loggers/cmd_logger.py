import logging
from typing import Any, Dict, Optional

from ...domain.interpolation.interfaces.base_logger import BaseLogger


class CMDLogger(BaseLogger):
    def __init__(self, name: str = "CMDLogger"):
        """
        Initialize the CMD logger.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        # Ensure handlers are not duplicated if logger is initialized multiple times
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_info(self, message: str) -> None:
        """Log an informational message."""
        self.logger.info(message)

    def log_error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)

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
        model_type: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        notes: Optional[str] = None,
        collection_name: Optional[str] = None,
        step: Optional[int] = None,  # Now matches BaseLogger
    ):
        """Log model details for CMD output. Ignores some W&B specific metadata."""
        log_message = f"Logging Model: '{name}'"
        if model_type:
            log_message += f", Type: {model_type}"
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
