from abc import ABC, abstractmethod

from ..entities.model_artifact import ModelArtifact


class BaseModelArtifactRepository(ABC):
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
    def load(
        self,
        estimator_type: str,
        version_id: str,
        mapping_direction: str = "inverse",
    ) -> ModelArtifact:
        """
        Retrieves a specific ModelArtifact entity by its unique ID.

        Args:
            estimator_type: The estimator family (e.g., 'gaussian_process_nd').
            version_id: The unique identifier of the specific model version directory.
            mapping_direction: Whether to fetch a forward or inverse artifact.

        Returns:
            The loaded ModelArtifact entity.

        Raises:
            FileNotFoundError: If the model with the specified ID is not found.
            IOError: For other loading errors (e.g., corrupted files).
        """
        pass

    @abstractmethod
    def get_all_versions(
        self, estimator_type: str, mapping_direction: str = "inverse"
    ) -> list[ModelArtifact]:
        """
        Retrieves all trained versions of a model based on its type.

        Args:
             estimator_type: The type of interpolation model (e.g., 'gaussian_process_nd')

        Returns:
            A list of ModelArtifact entities, sorted by version (highest first)
        """
        pass

    @abstractmethod
    def get_latest_version(
        self, estimator_type: str, mapping_direction: str = "inverse"
    ) -> ModelArtifact:
        """
        Retrieves the latest trained version of a model based on its type.

        Args:
             estimator_type: The type of interpolation model

        Returns:
            The ModelArtifact entity with the highest version
        """
        pass
