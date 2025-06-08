from abc import ABC, abstractmethod
from typing import Any


class BaseLogger(ABC):
    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        """Log evaluation or training metrics."""

    @abstractmethod
    def log_model(
        self,
        model: Any,
        name: str,
        model_type: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        notes: str | None = None,
        collection_name: str | None = None,
    ):
        """Log a model."""
