import datetime
import json
import os
import pickle
from pathlib import Path
from typing import Any

from .....shared.config import ROOT_PATH
from ....domain.model_management.entities.model_artifact import ModelArtifact
from ....domain.model_management.interfaces.base_repository import (
    BaseInterpolationModelRepository,
)


class DateTimeEncoder(json.JSONEncoder):
    """
    Custom JSONEncoder to serialize datetime objects to ISO 8601 format.
    """

    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        return super().default(obj)


class PickleFileHandler:
    """
    Handles the serialization and deserialization of model artifacts (using pickle)
    and their associated metadata (using JSON) to and from the file system.
    """

    def save(self, obj: Any, file_path: Path):
        """Saves a Python object to a specified file path using pickle."""
        try:
            with open(file_path, "wb") as f:
                pickle.dump(obj, f)
        except Exception as e:
            raise IOError(f"Failed to pickle object to {file_path}: {e}") from e

    def load(self, file_path: Path) -> Any:
        """Loads a Python object from a specified file path using pickle."""
        if not file_path.exists():
            raise FileNotFoundError(f"Artifact not found at {file_path}")
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise IOError(f"Failed to unpickle object from {file_path}: {e}") from e

    def save_metadata(self, file_path: Path, metadata_content: dict[str, Any]):
        """Saves model metadata as a JSON file to a specified path."""
        try:
            with open(file_path, "w") as f:
                json.dump(metadata_content, f, indent=4, cls=DateTimeEncoder)
        except Exception as e:
            raise IOError(f"Failed to save metadata to {file_path}: {e}") from e

    def load_metadata(self, file_path: Path) -> dict[str, Any]:
        """Loads model metadata from a JSON file at a specified path."""
        if not file_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {file_path}")
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            raise IOError(f"Failed to load metadata from {file_path}: {e}") from e


class PickleInterpolationModelRepository(BaseInterpolationModelRepository):
    """
    Manages the persistence of ModelArtifact entities, including version tracking,
    using the file system for storage. Each model version (identified by its unique ID)
    is stored in its own dedicated directory.
    """

    def __init__(self):
        self._base_model_storage_path = ROOT_PATH / "models"
        self._inverse_decision_mapper_handler = PickleFileHandler()
        self._base_model_storage_path.mkdir(
            parents=True, exist_ok=True
        )  # Ensure the base models directory exists

    def save(self, interpolator_model: ModelArtifact) -> None:
        """
        Saves a new ModelArtifact entity, representing a specific training version.
        A dedicated directory is created for this model version using its unique ID.

        Args:
            interpolator_model: The ModelArtifact entity to save. Its 'id' field
                                 is used to determine the storage location.
        """
        if not interpolator_model.id:
            raise ValueError("ModelArtifact entity must have a unique 'id' for saving.")

        # Create a subdirectory for the interpolator type
        interpolators_directory = (
            self._base_model_storage_path / interpolator_model.parameters.get("type")
        )

        interpolators_directory.mkdir(exist_ok=True)

        # Then create a directory for the unique model ID within the type directory
        interpolator_version_directory = (
            interpolators_directory
            / interpolator_model.trained_at.strftime("%Y-%m-%d_%H-%M-%S")
        )
        os.makedirs(interpolator_version_directory, exist_ok=True)

        # Define file paths within the model's dedicated directory
        mapper_artifact_path = (
            interpolator_version_directory / "inverse_decision_mapper.pkl"
        )
        decisions_normalizer_artifact_path = (
            interpolator_version_directory / "decisions_normalizer.pkl"
        )
        objecitves_normalizer_artifact_path = (
            interpolator_version_directory / "objectives_normalizer.pkl"
        )
        metadata_file_path = interpolator_version_directory / "metadata.json"

        # Save the inverse decision mapper instance, normalizers, and metadata
        self._inverse_decision_mapper_handler.save(
            interpolator_model.inverse_decision_mapper, mapper_artifact_path
        )
        self._inverse_decision_mapper_handler.save(
            interpolator_model.decisions_normalizer, decisions_normalizer_artifact_path
        )
        self._inverse_decision_mapper_handler.save(
            interpolator_model.objectives_normalizer,
            objecitves_normalizer_artifact_path,
        )

        model_metadata = interpolator_model.to_save_format()
        self._inverse_decision_mapper_handler.save_metadata(
            metadata_file_path, model_metadata
        )

    def load(self, interpolator_type: str, model_version_id: str) -> ModelArtifact:
        """
        Retrieves a specific ModelArtifact entity by its type and unique version ID.
        This is a direct lookup, which is much more efficient than a global search.

        Args:
            interpolator_type: The type of the interpolator (e.g., 'gaussian_process_nd').
            model_version_id: The unique identifier of the specific model version to load.

        Returns:
            The loaded ModelArtifact entity.

        Raises:
            FileNotFoundError: If the model version with the specified ID is not found.
            IOError: For other loading errors (e.g., corrupted files).
        """
        # Construct the path directly from the type and ID.
        interpolator_version_directory = (
            self._base_model_storage_path / interpolator_type / model_version_id
        )

        if not interpolator_version_directory.exists():
            raise FileNotFoundError(
                f"Model version with ID '{model_version_id}' for type '{interpolator_type}' not found at {interpolator_version_directory}"
            )

        metadata_file_path = interpolator_version_directory / "metadata.json"
        mapper_artifact_path = (
            interpolator_version_directory / "inverse_decision_mapper.pkl"
        )
        decisions_normalizer_artifact_path = (
            interpolator_version_directory / "decisions_normalizer.pkl"
        )
        objecitves_normalizer_artifact_path = (
            interpolator_version_directory / "objectives_normalizer.pkl"
        )

        # Load metadata
        model_metadata = self._inverse_decision_mapper_handler.load_metadata(
            metadata_file_path
        )

        # Load the fitted artifacts
        inverse_decision_mapper = self._inverse_decision_mapper_handler.load(
            mapper_artifact_path
        )
        decisions_normalizer = self._inverse_decision_mapper_handler.load(
            decisions_normalizer_artifact_path
        )
        objectives_normalizer = self._inverse_decision_mapper_handler.load(
            objecitves_normalizer_artifact_path
        )

        return ModelArtifact.from_saved_format(
            saved_data=model_metadata,
            inverse_decision_mapper=inverse_decision_mapper,
            objectives_normalizer=objectives_normalizer,
            decisions_normalizer=decisions_normalizer,
        )

    def get_all_versions(self, interpolator_type: str) -> list[ModelArtifact]:
        """
        Retrieves all trained versions of a model based on its 'type' from the parameters.

        Args:
            interpolator_type: The type of the interpolator model (e.g., 'gaussian_process_nd').

        Returns:
            A list of ModelArtifact entities, sorted by 'trained_at' timestamp in descending order (latest first).
        """
        found_model_versions: list[ModelArtifact] = []
        interpolators_directory = self._base_model_storage_path / interpolator_type
        if not interpolators_directory.exists():
            return []

        for interpolator_version_directory in interpolators_directory.iterdir():
            if interpolator_version_directory.is_dir():
                try:
                    # Pass the interpolator_type along with the directory name (the ID) to the load method.
                    model_version = self.load(
                        interpolator_type, interpolator_version_directory.name
                    )
                    found_model_versions.append(model_version)
                except (
                    FileNotFoundError,
                    json.JSONDecodeError,
                    KeyError,
                    ValueError,
                ) as e:
                    print(
                        f"Warning: Could not process directory '{interpolator_version_directory.name}': {e}. Skipping."
                    )
                    continue

        found_model_versions.sort(key=lambda model: model.trained_at, reverse=True)
        return found_model_versions

    def get_latest_version(self, interpolator_type: str) -> ModelArtifact:
        """
        Retrieves the latest trained version of a model based on its type.
        The 'latest' version is determined by the most recent 'trained_at' timestamp.

        Args:
            interpolator_type: The type of the interpolator model.

        Returns:
            The ModelArtifact entity representing the latest version.

        Raises:
            FileNotFoundError: If no model versions are found for the given type.
            Exception: For other errors during version lookup.
        """
        found_model_versions = self.get_all_versions(interpolator_type)

        if not found_model_versions:
            raise FileNotFoundError(
                f"No model versions found for type: '{interpolator_type}'"
            )

        # The list is already sorted by 'trained_at' in descending order, so the first element is the latest.
        return found_model_versions[0]
