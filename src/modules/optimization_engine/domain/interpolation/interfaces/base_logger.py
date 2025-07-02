from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseLogger(ABC):
    @abstractmethod
    def log_info(self, message: str) -> None:
        """Log an informational message."""
        pass

    @abstractmethod
    def log_error(self, message: str) -> None:
        """Log an error message."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log evaluation or training metrics."""
        pass

    @abstractmethod
    def log_model(
        self,
        model: Any,
        name: str,  # Model name/artifact name is crucial for tracking
        model_type: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        notes: Optional[str] = None,
        collection_name: Optional[str] = None,
        # The 'step' parameter should ideally be part of log_metrics or handle via model versioning,
        # but if a logger needs it, make it optional here too.
        step: Optional[int] = None,  # Added to support CMDLogger's original signature
    ):
        """Log a model with associated metadata."""
        pass
