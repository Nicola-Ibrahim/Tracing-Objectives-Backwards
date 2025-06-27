import logging
from typing import Any

from ...domain.interpolation.interfaces.base_logger import BaseLogger


class CMDLogger(BaseLogger):
    def __init__(self, name: str = "CMDLogger"):
        """
        Initialize the CMD logger.

        Args:
            name: Name of the logger.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        """Log evaluation or training metrics."""
        if step is not None:
            self.logger.info(f"Step {step}: {metrics}")
        else:
            self.logger.info(f"Metrics: {metrics}")

    def log_model(
        self,
        model: object,
        step: int | None = None,
        model_type: str = "model",
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        notes: str | None = None,
        collection_name: str | None = None,
    ):
        """Log model details."""
        if step is not None:
            self.logger.info(f"Step {step}: Model details: {model}")
        else:
            self.logger.info(f"Model details: {model}")
