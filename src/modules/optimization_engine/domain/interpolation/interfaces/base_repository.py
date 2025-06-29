from abc import ABC, abstractmethod

from ..entities.interpolator_model import InterpolatorModel


class BaseInterpolationModelRepository(ABC):
    """
    Abstract base class for a repository that handles persistence
    of InterpolatorModel entities, supporting version tracking.
    """

    @abstractmethod
    def save(self, interpolator_model: InterpolatorModel) -> None:
        """
        Saves a new InterpolatorModel entity (representing a specific training run/version).
        Each version is saved in a unique directory identified by its ID.

        Args:
            model_entity: The InterpolatorModel entity to save. Its 'id' field
                          will determine the storage location.
        """
        pass

    @abstractmethod
    def load(self, model_version_id: str) -> InterpolatorModel:
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
    def get_all_versions(self, interpolator_type: str) -> list[InterpolatorModel]:
        """
        Retrieves all trained versions of a model based on its type.

        Args:
            interpolator_type: The type of interpolation model (e.g., 'gaussian_process_nd')

        Returns:
            A list of InterpolatorModel entities, sorted by version_number (highest first)
        """
        pass

    @abstractmethod
    def get_latest_version(self, interpolator_type: str) -> InterpolatorModel:
        """
        Retrieves the latest trained version of a model based on its type.

        Args:
            interpolator_type: The type of interpolation model

        Returns:
            The InterpolatorModel entity with the highest version_number
        """
        pass
