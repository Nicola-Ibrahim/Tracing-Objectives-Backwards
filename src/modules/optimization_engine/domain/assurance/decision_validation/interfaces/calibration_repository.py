from abc import ABC, abstractmethod

from ..entities import DecisionValidationCalibration


class DecisionValidationCalibrationRepository(ABC):
    """Abstract repository for persisting decision-validation calibrations."""

    @abstractmethod
    def save(self, calibration: DecisionValidationCalibration) -> None: ...

    @abstractmethod
    def load_latest(self, scope: str) -> DecisionValidationCalibration: ...

    @abstractmethod
    def load(
        self, scope: str, calibration_id: str
    ) -> DecisionValidationCalibration: ...
