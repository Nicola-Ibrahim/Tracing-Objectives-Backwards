from abc import ABC, abstractmethod
from typing import Any

from ..aggregates.diagnostic_result import DiagnosticResult


class BaseDiagnosticRepository(ABC):
    """
    Abstract base class for a repository that handles persistence
    of DiagnosticResult entities, supporting sequential run tracking.
    """

    @abstractmethod
    def save(self, result: DiagnosticResult) -> int:
        """
        Persists a DiagnosticResult. Each run is assigned a sequential number
        within the scope of the estimator's version.

        Returns:
            The assigned sequential run number.
        """
        pass

    @abstractmethod
    def load(
        self,
        estimator_type: str,
        estimator_version: int,
        run_number: int,
        dataset_name: str,
        mapping_direction: str = "inverse",
    ) -> DiagnosticResult:
        """
        Loads a specific diagnostic evaluation run.

        Args:
            estimator_type: e.g., 'mdn'.
            estimator_version: Numeric version of the model.
            run_number: Sequential evaluation ID (1, 2, 3...).
            dataset_name: Environment identifier.
        """
        pass

    @abstractmethod
    def get_all_runs(
        self,
        estimator_type: str,
        estimator_version: int,
        dataset_name: str,
        mapping_direction: str = "inverse",
    ) -> list[DiagnosticResult]:
        """Fetches all evaluation runs for a specific model version."""
        pass

    @abstractmethod
    def get_latest_run(
        self,
        estimator_type: str,
        estimator_version: int,
        dataset_name: str,
        mapping_direction: str = "inverse",
    ) -> DiagnosticResult:
        """Fetches the most recent evaluation run."""
        pass

    @abstractmethod
    def get_batch(
        self,
        estimators: list[Any],
        dataset_name: str,
        mapping_direction: str = "inverse",
    ) -> dict[str, DiagnosticResult]:
        """
        Fetches multiple evaluation runs.
        Input is a list of requests (type, version, run_number).
        """
        pass
