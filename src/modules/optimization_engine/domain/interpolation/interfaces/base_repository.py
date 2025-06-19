from abc import ABC, abstractmethod

from ..entities.interpolator_model import InterpolatorModel


class BaseTrainedModelRepository(ABC):
    """
    Interface for a repository responsible for saving and retrieving
    trained InterpolatorModel entities.
    """

    @abstractmethod
    def save(self, model: InterpolatorModel) -> None:
        """Saves a trained InterpolatorModel entity."""
        pass

    @abstractmethod
    def get_by_id(self, model_id: str) -> InterpolatorModel:
        """Retrieves a trained InterpolatorModel entity by its ID."""
        pass

    # You might add methods like get_all, filter_by_type, etc.
