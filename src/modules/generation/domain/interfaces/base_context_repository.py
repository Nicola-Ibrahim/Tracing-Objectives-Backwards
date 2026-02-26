from abc import ABC, abstractmethod

from ..entities.generation_context import GenerationContext


class BaseContextRepository(ABC):
    """
    Abstract repository interface for persisting GenerationContext.
    """

    @abstractmethod
    def save(self, context: GenerationContext) -> None:
        """
        Persists a GenerationContext entity.
        """
        pass

    @abstractmethod
    def load(self, dataset_name: str) -> GenerationContext:
        """
        Loads a GenerationContext entity by dataset name.
        """
        pass
