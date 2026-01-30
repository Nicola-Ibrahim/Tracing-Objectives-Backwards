from abc import ABC, abstractmethod
from pathlib import Path

from ....shared.config import ROOT_PATH
from ..aggregates.diagnostic_result import DiagnosticResult


class BaseVisualizer(ABC):
    def __init__(self, save_path: Path | None = None):
        self.save_path = save_path or ROOT_PATH / "reports/figures"
        self.save_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def plot(self, results: list[DiagnosticResult]) -> None:
        """Visualize diagnostic results."""
        raise NotImplementedError("Subclasses must implement this method.")
