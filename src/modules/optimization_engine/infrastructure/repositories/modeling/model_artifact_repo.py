import json
from pathlib import Path

from .....shared.config import ROOT_PATH
from ....domain.modeling.entities.model_artifact import ModelArtifact
from ....domain.modeling.interfaces.base_repository import (
    BaseModelArtifactRepository,
)
from ...processing.files.json import JsonFileHandler
from ...processing.files.pickle import PickleFileHandler


class VersionManager:
    def __init__(self, file_handler: JsonFileHandler):
        self._file_handler = file_handler

    def get_next_version(self, model_directory: Path) -> int:
        """
        Determines the next sequential version number for a model type.
        """
        existing_versions = []
        if model_directory.exists():
            for entry in model_directory.iterdir():
                if entry.is_dir():
                    try:
                        metadata_path = entry / "metadata.json"
                        if metadata_path.exists():
                            meta = self._file_handler.load(metadata_path)
                            vn = meta.get("version")
                            if isinstance(vn, int):
                                existing_versions.append(vn)
                    except Exception:
                        continue

        return max(existing_versions) + 1 if existing_versions else 1


class FileSystemModelArtifactRepository(BaseModelArtifactRepository):
    """
    Manages the persistence of ModelArtifact entities using the file system.
    """

    def __init__(self):
        self._base_model_storage_path = ROOT_PATH / "models"
        self._pickel_file_handler = PickleFileHandler()
        self._json_file_handler = JsonFileHandler()
        self._version_manager = VersionManager(self._json_file_handler)
        self._base_model_storage_path.mkdir(parents=True, exist_ok=True)

    def save(self, model_artifact: ModelArtifact) -> None:
        if not model_artifact.id:
            raise ValueError("ModelArtifact entity must have a unique 'id' for saving.")

        estimator_type = model_artifact.parameters.get("type", "unknown")
        models_directory = self._base_model_storage_path / estimator_type
        models_directory.mkdir(parents=True, exist_ok=True)

        # Let the VersionManager handle the logic of finding the next version
        next_version = self._version_manager.get_next_version(models_directory)
        model_artifact.version = next_version

        # Create directory for this version
        dir_name = (
            f"v{next_version}-{model_artifact.trained_at.strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        model_artifact_version_directory = models_directory / dir_name
        model_artifact_version_directory.mkdir(exist_ok=True)

        # Define file paths
        estimator_path = model_artifact_version_directory / "estimator.pkl"
        metadata_path = model_artifact_version_directory / "metadata.json"

        # Save all components using their dedicated handlers
        self._pickel_file_handler.save(model_artifact.estimator, estimator_path)

        # Prepare and save metadata
        metadata = model_artifact.model_dump(
            exclude={
                "estimator",
            }
        )
        self._json_file_handler.save(metadata, metadata_path)

    def load(self, estimator_type: str, version_id: str) -> ModelArtifact:
        model_artifact_version_directory = (
            self._base_model_storage_path / estimator_type / version_id
        )
        if not model_artifact_version_directory.exists():
            raise FileNotFoundError(f"Model version '{version_id}' not found.")

        # Define file paths
        metadata_path = model_artifact_version_directory / "metadata.json"
        estimator_path = model_artifact_version_directory / "estimator.pkl"

        # Use the correct handler for each file type
        metadata = self._json_file_handler.load(metadata_path)
        estimator = self._pickel_file_handler.load(estimator_path)

        # Reconstruct the Pydantic model
        metrics_payload = metadata.get("metrics")
        if metrics_payload is None:
            metrics_payload = {
                "train_mertics": metadata.get("train_mertics", []),
                "test_metrics": metadata.get("test_metrics", []),
                "cv_scores": metadata.get("cv_scores", []),
            }

        return ModelArtifact.from_data(
            id=metadata.get("id"),
            parameters=metadata["parameters"],
            estimator=estimator,
            metrics=metrics_payload,
            loss_history=metadata.get("loss_history", {}),
            trained_at=metadata.get("trained_at"),
            version=metadata.get("version"),
        )

    def get_all_versions(self, estimator_type: str) -> list[ModelArtifact]:
        """
        Retrieves all trained versions of a model based on its 'type' from the parameters.

        Args:
             estimator_type: The type of the model model (e.g., 'gaussian_process_nd').

        Returns:
            A list of ModelArtifact entities, sorted by 'trained_at' timestamp in descending order (latest first).
        """
        found_model_versions: list[ModelArtifact] = []
        models_directory = self._base_model_storage_path / estimator_type
        if not models_directory.exists():
            return []

        for model_artifact_version_directory in models_directory.iterdir():
            if model_artifact_version_directory.is_dir():
                try:
                    # Pass the estimator_type along with the directory name (the ID) to the load method.
                    model_version = self.load(
                        estimator_type, model_artifact_version_directory.name
                    )
                    found_model_versions.append(model_version)
                except (
                    FileNotFoundError,
                    json.JSONDecodeError,
                    KeyError,
                    ValueError,
                ) as e:
                    print(
                        f"Warning: Could not process directory '{model_artifact_version_directory.name}': {e}. Skipping."
                    )
                    continue

        found_model_versions.sort(key=lambda model: model.trained_at, reverse=True)
        return found_model_versions

    def get_latest_version(self, estimator_type: str) -> ModelArtifact:
        """
        Retrieves the latest trained version of a model based on its type.
        The 'latest' version is determined by the most recent 'trained_at' timestamp.

        Args:
             estimator_type: The type of the model model.

        Returns:
            The ModelArtifact entity representing the latest version.

        Raises:
            FileNotFoundError: If no model versions are found for the given type.
            Exception: For other errors during version lookup.
        """
        found_model_versions = self.get_all_versions(estimator_type)

        if not found_model_versions:
            raise FileNotFoundError(
                f"No model versions found for type: '{ estimator_type}'"
            )

        # The list is already sorted by 'trained_at' in descending order, so the first element is the latest.
        return found_model_versions[0]
