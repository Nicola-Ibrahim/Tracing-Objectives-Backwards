from abc import ABC, abstractmethod

from ..entities.model_artifact import ModelArtifact


class BaseInterpolationModelRepository(ABC):
    """
    Abstract base class for a repository that handles persistence
    of ModelArtifact entities, supporting version tracking.
    """

    @abstractmethod
    def save(self, model_artifact: ModelArtifact) -> None:
        """
        Saves a new ModelArtifact entity (representing a specific training run/version).
        Each version is saved in a unique directory identified by its ID.

        Args:
            model_entity: The ModelArtifact entity to save. Its 'id' field
                          will determine the storage location.
        """
        pass

    @abstractmethod
    def load(self, model_version_id: str) -> ModelArtifact:
        """
        Retrieves a specific ModelArtifact entity by its unique ID.

        Args:
            model_id: The unique identifier of the specific model version to load.

        Returns:
            The loaded ModelArtifact entity.

        Raises:
            FileNotFoundError: If the model with the specified ID is not found.
            IOError: For other loading errors (e.g., corrupted files).
        """
        pass

    @abstractmethod
    def get_all_versions(self, model_type: str) -> list[ModelArtifact]:
        """
        Retrieves all trained versions of a model based on its type.

        Args:
             model_type: The type of interpolation model (e.g., 'gaussian_process_nd')

        Returns:
            A list of ModelArtifact entities, sorted by version (highest first)
        """
        pass

    @abstractmethod
    def get_latest_version(self, model_type: str) -> ModelArtifact:
        """
        Retrieves the latest trained version of a model based on its type.

        Args:
             model_type: The type of interpolation model

        Returns:
            The ModelArtifact entity with the highest version
        """
        pass
