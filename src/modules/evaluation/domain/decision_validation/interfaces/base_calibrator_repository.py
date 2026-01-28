from abc import ABC, abstractmethod
from typing import Any


class BaseDecisionValidationCalibrationRepository(ABC):
    """Abstract repository for persisting decision-validation calibrations."""

    @abstractmethod
    def save(self, calibration: Any) -> None: ...

    @abstractmethod
    def load(self) -> Any: ...
