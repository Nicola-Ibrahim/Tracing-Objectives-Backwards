from abc import ABC, abstractmethod
import numpy as np


class BaseOODCalibrator(ABC):
    @abstractmethod
    def fit(self, X_cal_norm: np.ndarray) -> None: ...

    @abstractmethod
    def transform(self, sample: np.ndarray) -> dict[str, float]: ...

    @abstractmethod
    def evaluate(self, sample: np.ndarray) -> tuple[bool, dict[str, float | bool], str]: ...

    @property
    @abstractmethod
    def threshold(self) -> float: ...
