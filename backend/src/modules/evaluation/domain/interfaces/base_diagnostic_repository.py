from abc import ABC, abstractmethod
from typing import Any

from ..aggregates.diagnostic_report import DiagnosticReport


class BaseDiagnosticRepository(ABC):
    """
    Abstract base class for a repository that handles persistence
    of DiagnosticReport entities, supporting sequential run tracking.
    """

    @abstractmethod
    def save(self, report: DiagnosticReport) -> int:
        """
        Persists a DiagnosticReport. Each run is assigned a sequential number
        within the scope of the engine's version.

        Returns:
            The assigned sequential run number.
        """
        pass

    @abstractmethod
    def load(
        self,
        engine_type: str,
        engine_version: int,
        run_number: int,
        dataset_name: str,
        mapping_direction: str = "inverse",
    ) -> DiagnosticReport:
        """
        Loads a specific diagnostic evaluation run.

        Args:
            engine_type: e.g., 'MDN'.
            engine_version: Numeric version of the model.
            run_number: Sequential evaluation ID (1, 2, 3...).
            dataset_name: Environment identifier.
        """
        pass

    @abstractmethod
    def get_all_runs(
        self,
        engine_type: str,
        engine_version: int,
        dataset_name: str,
        mapping_direction: str = "inverse",
    ) -> list[DiagnosticReport]:
        """Fetches all evaluation runs for a specific model version."""
        pass

    @abstractmethod
    def get_latest_run(
        self,
        engine_type: str,
        engine_version: int,
        dataset_name: str,
        mapping_direction: str = "inverse",
    ) -> DiagnosticReport:
        """Fetches the most recent evaluation run."""
        pass

    @abstractmethod
    def get_batch(
        self,
        engines: list[Any],
        dataset_name: str,
        mapping_direction: str = "inverse",
    ) -> dict[str, DiagnosticReport]:
        """
        Fetches multiple evaluation runs.
        Input is a list of requests (type, version, run_number).
        """
        pass
