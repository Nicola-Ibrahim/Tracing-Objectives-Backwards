from abc import ABC, abstractmethod

from ..entities.coherence_context import CoherenceContext


class BaseContextRepository(ABC):
    """
    Abstract repository interface for persisting CoherenceContext.
    """

    @abstractmethod
    def save(self, context: CoherenceContext) -> None:
        """
        Persists a CoherenceContext entity.
        """
        pass

    @abstractmethod
    def load(self, dataset_name: str) -> CoherenceContext:
        """
        Loads a CoherenceContext entity by dataset name.
        """
        pass
