from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseLogger(ABC):
    @abstractmethod
    def log_info(self, message: str) -> None:
        """Log an informational message."""
        pass

    @abstractmethod
    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        pass

    @abstractmethod
    def log_error(self, message: str) -> None:
        """Log an error message."""
        pass

    @abstractmethod
    def log_debug(self, message: str) -> None:
        """Log a debug message. These messages are typically for debugging purposes
        and may not be shown in production environments unless logging level is set to DEBUG."""
        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log evaluation or training metrics.
        This method is not abstract as some loggers might not support direct metric logging
        or might handle it differently, allowing concrete classes to override or do nothing.
        """
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
        step: Optional[int] = None,
    ):
        """Log a model with associated metadata."""
        pass
