import pickle
from datetime import datetime
from pathlib import Path

from ....shared.config import ROOT_PATH
from ....shared.infrastructure.processing.files.json import JsonFileHandler
from ....shared.infrastructure.processing.files.safetensors import (
    SafeTensorsFileHandler,
)
from ....shared.infrastructure.processing.files.toml import TomlFileHandler
from ...application.factories.estimator import EstimatorFactory
from ...application.registry import ESTIMATOR_PARAM_REGISTRY
from ...domain.entities.trained_pipeline import TrainedPipeline
from ...domain.enums.estimator_type import EstimatorTypeEnum
from ...domain.interfaces.base_estimator import BaseEstimator
from ...domain.interfaces.base_repository import BaseTrainedPipelineRepository
from ...domain.interfaces.base_transform import BaseTransformStep
from ...domain.value_objects.estimator_step import EstimatorStep, TrainingLog
from ...domain.value_objects.evaluation_result import EvaluationResult
from ...domain.value_objects.split_step import SplitConfig, SplitStep


class TransformStepRegistry:
    @staticmethod
    def get_step_class(step_type: str) -> type[BaseTransformStep]:
        if step_type == "normalization":
            from ..normalizers import NormalizationStep

            return NormalizationStep
        raise ValueError(f"Unknown transform step type: {step_type}")


class VersionManager:
    def __init__(self, toml_handler: TomlFileHandler):
        self._toml_handler = toml_handler

    def get_next_version(self, model_directory: Path) -> int:
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


class FileSystemTrainedPipelineRepository(BaseTrainedPipelineRepository):
    """
    Manages the persistence of TrainedPipeline entities using the file system.
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

    def save(self, pipeline: TrainedPipeline) -> None:
        if not pipeline.id:
            raise ValueError(
                "TrainedPipeline entity must have a unique 'id' for saving."
            )

        estimator_type = pipeline.model.config.type
        models_directory = self._compute_models_directory(
            estimator_type, pipeline.mapping_direction, pipeline.dataset_name
        )
        models_directory.mkdir(parents=True, exist_ok=True)

        next_version = pipeline.version or self._version_manager.get_next_version(
            models_directory
        )
        pipeline.version = next_version

        dir_name = (
            f"v{next_version}-{pipeline.trained_at.strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        pipeline_dir = models_directory / dir_name
        pipeline_dir.mkdir(exist_ok=True)

        # Basic metadata
        metadata = {
            "id": pipeline.id,
            "version": pipeline.version,
            "mapping_direction": pipeline.mapping_direction,
            "dataset_name": pipeline.dataset_name,
            "trained_at": pipeline.trained_at.isoformat(),
            "run_metadata": pipeline.run_metadata,
        }
        self._toml_file_handler.save(
            {"metadata": metadata}, pipeline_dir / "metadata.toml"
        )

        # Split Step
        split_dict = pipeline.split.dict()
        self._json_file_handler.save(split_dict, pipeline_dir / "split.json")

        # Transforms
        transforms_dir = pipeline_dir / "transforms"
        transforms_dir.mkdir(exist_ok=True)
        for idx, transform in enumerate(pipeline.transforms):
            t_dir = transforms_dir / f"{idx}_{transform.name}"
            t_dir.mkdir(exist_ok=True)
            self._json_file_handler.save(transform.config, t_dir / "config.json")
            with open(t_dir / "fitted_state.pkl", "wb") as f:
                pickle.dump(transform.get_fitted_state(), f)

        # Estimator Step
        estimator_dir = pipeline_dir / "estimator"
        estimator_dir.mkdir(exist_ok=True)
        self._json_file_handler.save(
            pipeline.model.config.model_dump(mode="json"), estimator_dir / "config.json"
        )
        self._json_file_handler.save(
            pipeline.model.training_log.model_dump(mode="json"),
            estimator_dir / "training_log.json",
        )

        checkpoint = pipeline.model.fitted.to_checkpoint()
        model_state = checkpoint.pop("model_state", None)
        if model_state:
            self._safetensors_handler.save(
                model_state, estimator_dir / "model_state.safetensors"
            )
        if checkpoint:
            # any architecture params from checkpoint
            self._json_file_handler.save(
                checkpoint, estimator_dir / "checkpoint_params.json"
            )

        # Evaluation
        self._json_file_handler.save(
            pipeline.evaluation.model_dump(), pipeline_dir / "evaluation.json"
        )

    def load(
        self,
        estimator_type: str,
        version_id: str,
        mapping_direction: str = "inverse",
        dataset_name: str | None = None,
    ) -> TrainedPipeline:
        models_directory = self._resolve_models_directory_for_read(
            estimator_type, mapping_direction, dataset_name
        )
        pipeline_dir = models_directory / version_id
        if not pipeline_dir.exists():
            raise FileNotFoundError(f"Pipeline version '{version_id}' not found.")

        # top level metadata
        metadata = self._toml_file_handler.load(pipeline_dir / "metadata.toml").get(
            "metadata", {}
        )

        # Load Split
        split_path = pipeline_dir / "split.json"
        if split_path.exists():
            split_data = self._json_file_handler.load(split_path)
            split = SplitStep(**split_data)
        else:
            split = SplitStep(config=SplitConfig())

        # Load Transforms
        transforms = []
        transforms_dir = pipeline_dir / "transforms"
        if transforms_dir.exists():
            subdirs = sorted(
                [d for d in transforms_dir.iterdir() if d.is_dir()],
                key=lambda x: int(x.name.split("_")[0]),
            )
            for d in subdirs:
                config = self._json_file_handler.load(d / "config.json")
                with open(d / "fitted_state.pkl", "rb") as f:
                    state = pickle.load(f)
                step_cls = TransformStepRegistry.get_step_class(config["type"])
                transforms.append(step_cls.from_fitted_state(config, state))

        # Load Estimator Step
        estimator_dir = pipeline_dir / "estimator"
        if estimator_dir.exists():
            config_data = self._json_file_handler.load(estimator_dir / "config.json")
            if "type" not in config_data:
                config_data["type"] = estimator_type

            checkpoint = {}
            chk_params = estimator_dir / "checkpoint_params.json"
            if chk_params.exists():
                checkpoint.update(self._json_file_handler.load(chk_params))
            state_path = estimator_dir / "model_state.safetensors"
            if state_path.exists():
                checkpoint["model_state"] = self._safetensors_handler.load(state_path)

            merged_params = {**config_data, **checkpoint}

            est_enum = EstimatorTypeEnum(config_data.get("type"))
            estimator_class = EstimatorFactory._registry.get(est_enum.value)
            fitted_estimator = estimator_class.from_checkpoint(merged_params)

            params_model = ESTIMATOR_PARAM_REGISTRY.get(est_enum)
            allowed = set(params_model.model_fields.keys())
            filtered = {k: v for k, v in config_data.items() if k in allowed}
            est_config = params_model.model_validate(filtered)

            hist_path = estimator_dir / "training_log.json"
            training_log = (
                TrainingLog(**self._json_file_handler.load(hist_path))
                if hist_path.exists()
                else TrainingLog()
            )

            estimator_step = EstimatorStep(
                config=est_config, fitted=fitted_estimator, training_log=training_log
            )
        else:
            # Fallback legacy loading here, but omitted for brevity, assuming only fresh artifacts for now, or minimal compat
            raise FileNotFoundError("Legacy fallback unimplemented")

        # Load Evaluation
        eval_path = pipeline_dir / "evaluation.json"
        if eval_path.exists():
            evaluation = EvaluationResult(**self._json_file_handler.load(eval_path))
        else:
            evaluation = EvaluationResult()

        return TrainedPipeline(
            id=metadata.get("id"),
            version=metadata.get("version"),
            dataset_name=metadata.get("dataset_name", dataset_name or "dataset"),
            mapping_direction=metadata.get("mapping_direction", mapping_direction),
            run_metadata=metadata.get("run_metadata", {}),
            trained_at=datetime.fromisoformat(metadata["trained_at"])
            if isinstance(metadata.get("trained_at"), str)
            else metadata.get("trained_at"),
            split=split,
            transforms=transforms,
            model=estimator_step,
            evaluation=evaluation,
        )

    def get_all_versions(
        self,
        estimator_type: str,
        mapping_direction: str = "inverse",
        dataset_name: str | None = None,
    ) -> list[TrainedPipeline]:
        found: list[TrainedPipeline] = []
        models_directory = self._resolve_models_directory_for_read(
            estimator_type, mapping_direction, dataset_name
        )
        if not models_directory.exists():
            return []

        for p_dir in models_directory.iterdir():
            if p_dir.is_dir():
                try:
                    p = self.load(
                        estimator_type,
                        p_dir.name,
                        mapping_direction=mapping_direction,
                        dataset_name=dataset_name,
                    )
                    found.append(p)
                except Exception as e:
                    print(f"Warning: Could not process '{p_dir.name}': {e}")
                    continue

        found.sort(key=lambda x: x.trained_at, reverse=True)
        return found

    def get_latest_version(
        self,
        estimator_type: str,
        mapping_direction: str = "inverse",
        dataset_name: str | None = None,
    ) -> TrainedPipeline:
        found = self.get_all_versions(estimator_type, mapping_direction, dataset_name)
        if not found:
            raise FileNotFoundError(
                f"No model versions found for type '{estimator_type}' and direction '{mapping_direction}'"
            )
        return found[0]

    def get_version_by_number(
        self,
        estimator_type: str,
        version: int,
        mapping_direction: str = "inverse",
        dataset_name: str | None = None,
    ) -> TrainedPipeline:
        models_directory = self._resolve_models_directory_for_read(
            estimator_type, mapping_direction, dataset_name
        )
        version_dir = self._find_version_dir(models_directory, version)
        if not version_dir:
            raise FileNotFoundError(f"Version {version} not found")
        return self.load(
            estimator_type, version_dir.name, mapping_direction, dataset_name
        )

    def get_estimators(
        self,
        *,
        mapping_direction: str,
        requested: list[tuple[str, int | None]],
        dataset_name: str | None = None,
        on_missing: str = "skip",
    ) -> list[tuple[str, BaseEstimator]]:
        resolved: list[tuple[str, BaseEstimator]] = []
        for estimator_type, version in requested:
            try:
                if version is None:
                    p = self.get_latest_version(
                        estimator_type, mapping_direction, dataset_name
                    )
                    name = f"{estimator_type} (Latest)"
                else:
                    p = self.get_version_by_number(
                        estimator_type, version, mapping_direction, dataset_name
                    )
                    name = f"{estimator_type} (v{version})"
                resolved.append((name, p.model.fitted))
            except FileNotFoundError:
                if on_missing == "raise":
                    raise
        return resolved
