from abc import ABC, abstractmethod

from ..entities.interpolator_model import TrainedModelArtifact


class BaseInterpolationModelRepository(ABC):
    """
    Abstract base class for a repository that handles persistence
    of TrainedModelArtifact entities, supporting version tracking.
    """

    @abstractmethod
    def save(self, interpolator_model: TrainedModelArtifact) -> None:
        """
        Saves a new TrainedModelArtifact entity (representing a specific training run/version).
        Each version is saved in a unique directory identified by its ID.

        Args:
            model_entity: The TrainedModelArtifact entity to save. Its 'id' field
                          will determine the storage location.
        """
        pass

    @abstractmethod
    def load(self, model_version_id: str) -> TrainedModelArtifact:
        """
        Retrieves a specific TrainedModelArtifact entity by its unique ID.

        Args:
            model_id: The unique identifier of the specific model version to load.

        Returns:
            The loaded TrainedModelArtifact entity.

        Raises:
            FileNotFoundError: If the model with the specified ID is not found.
            IOError: For other loading errors (e.g., corrupted files).
        """
        pass

    @abstractmethod
    def get_all_versions(self, interpolator_type: str) -> list[TrainedModelArtifact]:
        """
        Retrieves all trained versions of a model based on its type.

        Args:
            interpolator_type: The type of interpolation model (e.g., 'gaussian_process_nd')

        Returns:
            A list of TrainedModelArtifact entities, sorted by version_number (highest first)
        """
        pass

    @abstractmethod
    def get_latest_version(self, interpolator_type: str) -> TrainedModelArtifact:
        """
        Retrieves the latest trained version of a model based on its type.

        Args:
            interpolator_type: The type of interpolation model

        Returns:
            The TrainedModelArtifact entity with the highest version_number
        """
        pass
