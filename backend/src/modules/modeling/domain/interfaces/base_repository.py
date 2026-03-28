from abc import ABC, abstractmethod

from ..entities.trained_pipeline import TrainedPipeline
from .base_estimator import BaseEstimator


class BaseTrainedPipelineRepository(ABC):
    """
    Abstract base class for a repository that handles persistence
    of TrainedPipeline entities, supporting version tracking.
    """

    @abstractmethod
    def save(self, pipeline: TrainedPipeline) -> None:
        """
        Saves a new TrainedPipeline entity representing a training run.
        Each version is saved in a unique directory identified by its ID.

        Args:
            pipeline: The TrainedPipeline entity to save. Its 'id' field
                      will determine the storage location.
        """
        pass

    @abstractmethod
    def load(
        self,
        estimator_type: str,
        version_id: str,
        mapping_direction: str = "inverse",
        dataset_name: str | None = None,
    ) -> TrainedPipeline:
        """
        Retrieves a specific TrainedPipeline entity by its unique ID.

        Args:
            estimator_type: The estimator family (e.g., 'gaussian_process_nd').
            version_id: The unique identifier of the specific model version directory.
            mapping_direction: Whether to fetch a forward or inverse artifact.

        Returns:
            The loaded TrainedPipeline entity.

        Raises:
            FileNotFoundError: If the model with the specified ID is not found.
            IOError: For other loading errors (e.g., corrupted files).
        """
        pass

    @abstractmethod
    def get_all_versions(
        self,
        estimator_type: str,
        mapping_direction: str = "inverse",
        dataset_name: str | None = None,
    ) -> list[TrainedPipeline]:
        """
        Retrieves all trained versions of a model based on its type.

        Args:
             estimator_type: Interpolation model (e.g., 'gaussian_process_nd')

        Returns:
            A list of TrainedPipeline entities, sorted by version (highest first)
        """
        pass

    @abstractmethod
    def get_latest_version(
        self,
        estimator_type: str,
        mapping_direction: str = "inverse",
        dataset_name: str | None = None,
    ) -> TrainedPipeline:
        """
        Retrieves the latest trained version of a model based on its type.

        Args:
             estimator_type: The type of interpolation model

        Returns:
            The TrainedPipeline entity with the highest version
        """
        pass

    @abstractmethod
    def get_version_by_number(
        self,
        estimator_type: str,
        version: int,
        mapping_direction: str = "inverse",
        dataset_name: str | None = None,
    ) -> TrainedPipeline:
        """
        Retrieves a specific model version by its numeric version field.

        Args:
             estimator_type: The type of interpolation model
             version: The numeric version to load.
        """
        pass

    @abstractmethod
    def get_estimators(
        self,
        *,
        mapping_direction: str,
        requested: list[tuple[str, int | None]],
        dataset_name: str | None = None,
        on_missing: str = "skip",
    ) -> list[tuple[str, BaseEstimator]]:
        """Resolve estimators for the requested (type, version) pairs.

        Args:
            mapping_direction: "inverse" or "forward".
            requested: List of (type, version) pairs. If version is None, 
                       latest is used.
            on_missing: "skip" (default) skips missing; "raise" raises on first.

        Returns:
            List of (display_name, estimator) pairs in the same order as requested.
        """
        raise NotImplementedError
