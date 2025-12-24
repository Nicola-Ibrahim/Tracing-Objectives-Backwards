import json
from datetime import datetime
from pathlib import Path

from .....shared.config import ROOT_PATH
from ....application.factories.estimator import EstimatorFactory
from ....application.training.registry import ESTIMATOR_PARAM_REGISTRY
from ....domain.modeling.entities.model_artifact import ModelArtifact
from ....domain.modeling.enums.estimator_type import EstimatorTypeEnum
from ....domain.modeling.interfaces.base_estimator import BaseEstimator
from ....domain.modeling.interfaces.base_repository import (
    BaseModelArtifactRepository,
)
from ....domain.modeling.value_objects.estimator_params import EstimatorParamsBase
from ...processing.files.json import JsonFileHandler
from ...processing.files.safetensors import SafeTensorsFileHandler
from ...processing.files.toml import TomlFileHandler


class VersionManager:
    def __init__(self, toml_handler: TomlFileHandler):
        self._toml_handler = toml_handler

    def get_next_version(self, model_directory: Path) -> int:
        """
        Determines the next sequential version number for a model type.
        """
        existing_versions = []
        if model_directory.exists():
            for entry in model_directory.iterdir():
                if entry.is_dir():
                    try:
                        metadata_toml = entry / "metadata.toml"
                        if metadata_toml.exists():
                            meta = self._toml_handler.load(metadata_toml)
                            vn = meta.get("metadata", {}).get("version")
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
        self._toml_file_handler = TomlFileHandler()
        self._safetensors_handler = SafeTensorsFileHandler()
        self._version_manager = VersionManager(self._toml_file_handler)
        self._base_model_storage_path.mkdir(parents=True, exist_ok=True)

    def _compute_models_directory(
        self, estimator_type: str, mapping_direction: str, dataset_name: str
    ) -> Path:
        return (
            self._base_model_storage_path
            / mapping_direction
            / dataset_name
            / estimator_type
        )

    def _resolve_models_directory_for_read(
        self, estimator_type: str, mapping_direction: str, dataset_name: str | None
    ) -> Path:
        """
        Resolve the directory where artifacts are stored. For inverse models we
        keep a fallback to the legacy layout (<models>/<estimator_type>).
        """
        dataset_key = dataset_name or "dataset"
        preferred = self._compute_models_directory(
            estimator_type, mapping_direction, dataset_key
        )
        if preferred.exists():
            return preferred

        legacy_mapping = (
            self._base_model_storage_path / mapping_direction / estimator_type
        )
        if legacy_mapping.exists():
            return legacy_mapping

        if mapping_direction == "inverse":
            legacy = self._base_model_storage_path / estimator_type
            if legacy.exists():
                return legacy

        return preferred

    @staticmethod
    def _find_version_dir(models_directory: Path, version: int) -> Path | None:
        if not models_directory.exists():
            return None
        prefix = f"v{version}-"
        for entry in models_directory.iterdir():
            if entry.is_dir() and entry.name.startswith(prefix):
                return entry
        return None

    def save(self, model_artifact: ModelArtifact) -> None:
        if not model_artifact.id:
            raise ValueError("ModelArtifact entity must have a unique 'id' for saving.")

        estimator_type = model_artifact.parameters.type
        mapping_direction = model_artifact.mapping_direction
        dataset_name = model_artifact.dataset_name
        models_directory = self._compute_models_directory(
            estimator_type, mapping_direction, dataset_name
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

        # Use command-provided estimator parameters for hyperparameter persistence
        serialized_hyperparams = model_artifact.parameters.model_dump(mode="json")

        # Merge hyperparameters and checkpoint into parameters (flattened structure)
        # BUT extract training_history to ensure it's removed from checkpoint if it was there
        checkpoint.pop("training_history", None)
        model_state = checkpoint.pop("model_state", None)
        merged_parameters = {**serialized_hyperparams, **checkpoint}
        merged_parameters.setdefault("type", model_artifact.parameters.type)
        merged_parameters["mapping_direction"] = mapping_direction

        # Build metadata without estimator/params/training history
        metadata = model_artifact.model_dump(
            exclude={"estimator", "parameters", "training_history", "metrics"}
        )
        metadata_toml = {
            "metadata": metadata,
            "parameters": merged_parameters,
        }
        training_history_path = (
            model_artifact_version_directory / "training_history.json"
        )
        training_payload = dict(model_artifact.training_history)
        training_payload["metrics"] = model_artifact.metrics.model_dump()
        self._json_file_handler.save(training_payload, training_history_path)

        if model_state:
            state_path = model_artifact_version_directory / "model_state.safetensors"
            self._safetensors_handler.save(model_state, state_path)
        metadata_path = model_artifact_version_directory / "metadata.toml"
        self._toml_file_handler.save(metadata_toml, metadata_path)

    def load(
        self,
        estimator_type: str,
        version_id: str,
        mapping_direction: str = "inverse",
        dataset_name: str | None = None,
    ) -> ModelArtifact:
        models_directory = self._resolve_models_directory_for_read(
            estimator_type, mapping_direction, dataset_name
        )
        model_artifact_version_directory = models_directory / version_id
        if not model_artifact_version_directory.exists():
            raise FileNotFoundError(f"Model version '{version_id}' not found.")

        metadata_toml_path = model_artifact_version_directory / "metadata.toml"
        loaded_training_history = None
        loaded_metrics = None
        if metadata_toml_path.exists():
            payload = self._toml_file_handler.load(metadata_toml_path)
            metadata = payload.get("metadata", {})
            parameters = payload.get("parameters", {})
        else:
            metadata_path = model_artifact_version_directory / "metadata.json"
            metadata = self._json_file_handler.load(metadata_path)
            parameters = {}
            params_file = metadata.get("parameters_file")
            if params_file:
                params_path = model_artifact_version_directory / params_file
                parameters.update(self._toml_file_handler.load(params_path))
            elif "parameters" in metadata:
                parameters.update(metadata["parameters"])
            loaded_training_history = None
            history_file = metadata.get("training_history_file")
            if history_file:
                history_path = model_artifact_version_directory / history_file
                if history_path.exists():
                    loaded_training_history = self._json_file_handler.load(history_path)
        parameters = dict(parameters)
        dataset_from_params = parameters.pop("dataset_name", None)
        if "type" not in parameters and estimator_type:
            parameters["type"] = estimator_type

        # Check if this is new format (checkpoint merged into parameters)
        # or old format (separate checkpoint field)
        if "checkpoint" in metadata:
            # Old format - merge checkpoint into parameters for compatibility
            parameters.update(metadata["checkpoint"])

        # Add state dict from safetensors if not already present
        if "model_state" not in parameters:
            state_file = metadata.get("state_file", "model_state.safetensors")
            state_path = model_artifact_version_directory / state_file
            if state_path.exists():
                parameters["model_state"] = self._safetensors_handler.load(state_path)

        # Add training_history from file or root level if present
        if loaded_training_history is None:
            history_path = model_artifact_version_directory / "training_history.json"
            if history_path.exists():
                history_payload = self._json_file_handler.load(history_path)
                loaded_metrics = history_payload.pop("metrics", None)
                loaded_training_history = history_payload
        if loaded_training_history is None and "training_history" in metadata:
            loaded_training_history = metadata["training_history"]
        if loaded_training_history is not None:
            parameters["training_history"] = loaded_training_history

        # Look up estimator class from registry
        try:
            estimator_type = EstimatorTypeEnum(parameters.get("type"))
        except ValueError as exc:
            raise ValueError(
                f"Unknown estimator type: {parameters.get('type')}. "
                f"Available types: {[e.value for e in EstimatorTypeEnum]}"
            ) from exc
        estimator_class = EstimatorFactory._registry.get(estimator_type.value)
        if not estimator_class:
            raise ValueError(
                f"Unknown estimator type: {parameters.get('type')}. "
                f"Available types: {list(EstimatorFactory._registry.keys())}"
            )

        # Rebuild estimator from parameters (which now contains checkpoint data)
        estimator = estimator_class.from_checkpoint(parameters)

        # Reconstruct the Pydantic model
        metrics_payload = loaded_metrics or metadata.get("metrics")
        if metrics_payload is None:
            metrics_payload = {
                "train_mertics": metadata.get("train_mertics", []),
                "test_metrics": metadata.get("test_metrics", []),
                "cv_scores": metadata.get("cv_scores", []),
            }

        # Unified training_history from root level
        training_history = loaded_training_history
        # Fallback for old loss_history if training_history is missing
        if training_history is None:
            old_loss = metadata.get("loss_history", {})
            training_history = {
                "epochs": old_loss.get("bins", []),
                "train_loss": old_loss.get("train_loss", []),
                "val_loss": old_loss.get("val_loss", []),
            }

        params_model = ESTIMATOR_PARAM_REGISTRY.get(parameters.get("type"))
        if params_model is None:
            raise ValueError(
                f"Unsupported estimator params type: {parameters.get('type')!r}"
            )
        allowed = set(params_model.model_fields.keys())
        filtered = {k: v for k, v in parameters.items() if k in allowed}
        estimator_params: EstimatorParamsBase = params_model.model_validate(filtered)
        resolved_mapping_direction = metadata.get(
            "mapping_direction", parameters.get("mapping_direction", "inverse")
        )
        resolved_dataset_name = metadata.get(
            "dataset_name", dataset_from_params or "dataset"
        )
        run_metadata = metadata.get("run_metadata", {})

        return ModelArtifact.from_data(
            id=metadata.get("id"),
            parameters=estimator_params,
            estimator=estimator,
            metrics=metrics_payload,
            training_history=training_history,
            trained_at=datetime.fromisoformat(metadata["trained_at"])
            if isinstance(metadata.get("trained_at"), str)
            else metadata.get("trained_at"),
            version=metadata.get("version"),
            mapping_direction=resolved_mapping_direction,
            dataset_name=resolved_dataset_name,
            run_metadata=run_metadata,
        )

    def get_all_versions(
        self,
        estimator_type: str,
        mapping_direction: str = "inverse",
        dataset_name: str | None = None,
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
            estimator_type, mapping_direction, dataset_name
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
                        dataset_name=dataset_name,
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
        self,
        estimator_type: str,
        mapping_direction: str = "inverse",
        dataset_name: str | None = None,
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
            estimator_type,
            mapping_direction=mapping_direction,
            dataset_name=dataset_name,
        )

        if not found_model_versions:
            raise FileNotFoundError(
                f"No model versions found for type: '{estimator_type}' "
                f"and mapping_direction: '{mapping_direction}'"
            )

        # The list is already sorted by 'trained_at' in descending order, so the first element is the latest.
        return found_model_versions[0]

    def get_version_by_number(
        self,
        estimator_type: str,
        version: int,
        mapping_direction: str = "inverse",
        dataset_name: str | None = None,
    ) -> ModelArtifact:
        models_directory = self._resolve_models_directory_for_read(
            estimator_type, mapping_direction, dataset_name
        )
        version_dir = self._find_version_dir(models_directory, version)
        if not version_dir:
            raise FileNotFoundError(
                f"Version {version} not found for type: '{estimator_type}' "
                f"and mapping_direction: '{mapping_direction}'"
            )
        return self.load(
            estimator_type=estimator_type,
            version_id=version_dir.name,
            mapping_direction=mapping_direction,
            dataset_name=dataset_name,
        )

    def get_estimators(
        self,
        *,
        mapping_direction: str,
        requested: list[tuple[str, int | None]],
        dataset_name: str | None = None,
    ) -> list[tuple[str, BaseEstimator]]:
        """Resolve estimators by (type, version) pairs.

        This keeps estimator selection logic inside the repository, so handlers don't need
        to understand artifact directories or version lookups.

        Args:
            mapping_direction: "inverse" or "forward".
            requested: List of (estimator_type, version) tuples. Version can be None for latest.
            on_missing: "skip" to ignore missing estimators, "raise" to throw an error.

        Returns:
            List of (display_name, estimator) tuples for resolved estimators.
        """
        resolved: list[tuple[str, BaseEstimator]] = []

        for estimator_type, version in requested:
            if version is None:
                artifact = self.get_latest_version(
                    estimator_type=estimator_type,
                    mapping_direction=mapping_direction,
                    dataset_name=dataset_name,
                )
                display_name = f"{estimator_type} (Latest)"

            else:
                artifact = self.get_version_by_number(
                    estimator_type=estimator_type,
                    version=version,
                    mapping_direction=mapping_direction,
                    dataset_name=dataset_name,
                )
                display_name = f"{estimator_type} (v{version})"

            resolved.append((display_name, artifact.estimator))

        return resolved
