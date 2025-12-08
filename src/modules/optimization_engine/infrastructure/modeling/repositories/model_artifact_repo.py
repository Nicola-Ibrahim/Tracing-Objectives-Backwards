import json
from enum import Enum
from pathlib import Path

from .....shared.config import ROOT_PATH
from ....domain.modeling.entities.model_artifact import ModelArtifact
from ....domain.modeling.interfaces.base_repository import (
    BaseModelArtifactRepository,
)
from ....domain.modeling.value_objects.loss_history import LossHistory
from ...processing.files.json import JsonFileHandler
from ..ml.deterministic.coco_biobj_function import COCOEstimator

# Import estimators for registry
from ..ml.probabilistic.cvae import CVAEEstimator
from ..ml.probabilistic.inn import INNEstimator
from ..ml.probabilistic.mdn import MDNEstimator

# Estimator registry for loading from checkpoint
ESTIMATOR_REGISTRY = {
    "mdn": MDNEstimator,
    "cvae": CVAEEstimator,
    "inn": INNEstimator,
    "coco": COCOEstimator,
}


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
        self._json_file_handler = JsonFileHandler()
        self._version_manager = VersionManager(self._json_file_handler)
        self._base_model_storage_path.mkdir(parents=True, exist_ok=True)

    def _compute_models_directory(
        self, estimator_type: str, mapping_direction: str
    ) -> Path:
        return self._base_model_storage_path / mapping_direction / estimator_type

    def _resolve_models_directory_for_read(
        self, estimator_type: str, mapping_direction: str
    ) -> Path:
        """
        Resolve the directory where artifacts are stored. For inverse models we
        keep a fallback to the legacy layout (<models>/<estimator_type>).
        """
        preferred = self._compute_models_directory(estimator_type, mapping_direction)
        if preferred.exists():
            return preferred

        if mapping_direction == "inverse":
            legacy = self._base_model_storage_path / estimator_type
            if legacy.exists():
                return legacy

        return preferred

    def save(self, model_artifact: ModelArtifact) -> None:
        if not model_artifact.id:
            raise ValueError("ModelArtifact entity must have a unique 'id' for saving.")

        estimator_type = model_artifact.parameters.get("type", "unknown")
        mapping_direction = model_artifact.parameters.get(
            "mapping_direction", "inverse"
        )
        models_directory = self._compute_models_directory(
            estimator_type, mapping_direction
        )
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

        # Get checkpoint from estimator (contains model_state, dimensions, etc.)
        checkpoint = model_artifact.estimator.to_checkpoint()

        # Get ALL init hyperparameters (including private ones like _hidden_layers)
        hyperparams = model_artifact.estimator._collect_init_params_from_instance()

        # Serialize enums to strings
        serialized_hyperparams = {}
        for k, v in hyperparams.items():
            if isinstance(v, Enum):
                serialized_hyperparams[k] = v.value
            else:
                serialized_hyperparams[k] = v

        # Merge hyperparameters and checkpoint into parameters (flattened structure)
        # BUT extract training_history to put at root level
        training_history = checkpoint.pop("training_history", None)
        merged_parameters = {**serialized_hyperparams, **checkpoint}
        merged_parameters["type"] = model_artifact.parameters.get("type")
        merged_parameters["mapping_direction"] = mapping_direction

        # Build metadata without estimator and loss_history
        metadata = model_artifact.model_dump(
            exclude={"estimator", "loss_history", "parameters"}
        )
        metadata["parameters"] = merged_parameters  # Use merged params
        if training_history:
            metadata["training_history"] = training_history  # At root level

        # Save only metadata.json
        metadata_path = model_artifact_version_directory / "metadata.json"
        self._json_file_handler.save(metadata, metadata_path)

    def load(
        self,
        estimator_type: str,
        version_id: str,
        mapping_direction: str = "inverse",
    ) -> ModelArtifact:
        models_directory = self._resolve_models_directory_for_read(
            estimator_type, mapping_direction
        )
        model_artifact_version_directory = models_directory / version_id
        if not model_artifact_version_directory.exists():
            raise FileNotFoundError(f"Model version '{version_id}' not found.")

        # Load metadata.json
        metadata_path = model_artifact_version_directory / "metadata.json"
        metadata = self._json_file_handler.load(metadata_path)

        parameters = metadata["parameters"]

        # Check if this is new format (checkpoint merged into parameters)
        # or old format (separate checkpoint field)
        if "checkpoint" in metadata:
            # Old format - merge checkpoint into parameters for compatibility
            parameters.update(metadata["checkpoint"])

        # Add training_history from root level if present
        if "training_history" in metadata:
            parameters["training_history"] = metadata["training_history"]

        # Look up estimator class from registry
        estimator_class = ESTIMATOR_REGISTRY.get(parameters.get("type"))
        if not estimator_class:
            raise ValueError(
                f"Unknown estimator type: {parameters.get('type')}. "
                f"Available types: {list(ESTIMATOR_REGISTRY.keys())}"
            )

        # Rebuild estimator from parameters (which now contains checkpoint data)
        estimator = estimator_class.from_checkpoint(parameters)

        # Reconstruct the Pydantic model
        metrics_payload = metadata.get("metrics")
        if metrics_payload is None:
            metrics_payload = {
                "train_mertics": metadata.get("train_mertics", []),
                "test_metrics": metadata.get("test_metrics", []),
                "cv_scores": metadata.get("cv_scores", []),
            }

        # Use loss_history from metadata (should be LossHistory compatible)
        loss_history_data = metadata.get("loss_history", {})
        loss_history = (
            LossHistory(**loss_history_data) if loss_history_data else LossHistory()
        )

        return ModelArtifact.from_data(
            id=metadata.get("id"),
            parameters=metadata["parameters"],
            estimator=estimator,
            metrics=metrics_payload,
            loss_history=loss_history,
            trained_at=metadata.get("trained_at"),
            version=metadata.get("version"),
        )

    def get_all_versions(
        self, estimator_type: str, mapping_direction: str = "inverse"
    ) -> list[ModelArtifact]:
        """
        Retrieves all trained versions of a model based on its 'type' from the parameters.

        Args:
             estimator_type: The type of the model model (e.g., 'gaussian_process_nd').

        Returns:
            A list of ModelArtifact entities, sorted by 'trained_at' timestamp in descending order (latest first).
        """
        found_model_versions: list[ModelArtifact] = []
        models_directory = self._resolve_models_directory_for_read(
            estimator_type, mapping_direction
        )
        if not models_directory.exists():
            return []

        for model_artifact_version_directory in models_directory.iterdir():
            if model_artifact_version_directory.is_dir():
                try:
                    # Pass the estimator_type along with the directory name (the ID) to the load method.
                    model_version = self.load(
                        estimator_type,
                        model_artifact_version_directory.name,
                        mapping_direction=mapping_direction,
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

    def get_latest_version(
        self, estimator_type: str, mapping_direction: str = "inverse"
    ) -> ModelArtifact:
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
        found_model_versions = self.get_all_versions(
            estimator_type, mapping_direction=mapping_direction
        )

        if not found_model_versions:
            raise FileNotFoundError(
                f"No model versions found for type: '{estimator_type}' "
                f"and mapping_direction: '{mapping_direction}'"
            )

        # The list is already sorted by 'trained_at' in descending order, so the first element is the latest.
        return found_model_versions[0]
