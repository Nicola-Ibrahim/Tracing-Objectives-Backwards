from abc import ABC, abstractmethod
from typing import TypeVar

from ..entities.interpolator_model import InterpolatorModel

T = TypeVar("T", bound=InterpolatorModel)


class BaseInterpolationModelRepository(ABC):
    """
    Abstract base class for a repository that handles persistence
    of InterpolatorModel entities, supporting version tracking.
    """

    @abstractmethod
    def save(self, model_entity: T) -> None:
        """
        Saves a new InterpolatorModel entity (representing a specific training run/version).
        Each version is saved in a unique directory identified by its ID.

        Args:
            model_entity: The InterpolatorModel entity to save. Its 'id' field
                          will determine the storage location.
        """
        pass

    @abstractmethod
    def load(self, model_id: str) -> T:
        """
        Retrieves a specific InterpolatorModel entity by its unique ID.

        Args:
            model_id: The unique identifier of the specific model version to load.

        Returns:
            The loaded InterpolatorModel entity.

        Raises:
            FileNotFoundError: If the model with the specified ID is not found.
            IOError: For other loading errors (e.g., corrupted files).
        """
        pass

    @abstractmethod
    def get_latest_version(self, model_conceptual_name: str) -> T:
        """
        Retrieves the latest trained version of a model based on its conceptual name.
        The 'latest' version is determined by the 'trained_at' timestamp.

        Args:
            model_conceptual_name: The conceptual name of the model type (e.g., 'f1_vs_f2_PchipMapper').

        Returns:
            The InterpolatorModel entity representing the latest version.

        Raises:
            FileNotFoundError: If no model with the given conceptual name is found.
            Exception: For other errors during version lookup.
        """
        pass

    @abstractmethod
    def get_all_versions_by_conceptual_name(
        self, model_conceptual_name: str
    ) -> list[T]:
        """
        Retrieves all trained versions of a model based on its conceptual name.

        Args:
            model_conceptual_name: The conceptual name of the model type.

        Returns:
            A list of InterpolatorModel entities, sorted by trained_at timestamp (latest first).
        """
        pass
