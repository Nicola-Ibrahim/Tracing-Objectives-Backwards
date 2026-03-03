from abc import ABC, abstractmethod

from ..entities.inverse_mapping_engine import InverseMappingEngine


class BaseInverseMappingEngineRepository(ABC):
    """
    Abstract repository interface for persisting InverseMappingEngine.
    """

    @abstractmethod
    def save(self, engine: InverseMappingEngine) -> None:
        """
        Persists an InverseMappingEngine entity.
        Calculates version automatically based on existing ones.
        """
        pass

    @abstractmethod
    def load(
        self, dataset_name: str, solver_type: str, version: int | None = None
    ) -> InverseMappingEngine:
        """
        Loads an InverseMappingEngine entity by dataset name, solver type and optional version.
        If version is None, the most recently created version is returned.
        """
        pass

    @abstractmethod
    def list(self, dataset_name: str, solver_type: str | None = None) -> list[dict]:
        """
        Lists summaries of persisted engines for a given dataset and optional solver type.
        Returns a list of dictionaries with metadata (version, solver_type, created_at, etc.).
        """
        pass

    @abstractmethod
    def get_version_by_number(
        self, dataset_name: str, solver_type: str, version: int
    ) -> InverseMappingEngine:
        """
        Loads a specific engine version by its number.
        """
        pass
