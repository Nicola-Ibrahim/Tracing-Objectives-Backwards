from datetime import datetime
from pathlib import Path

import numpy as np

from ....modeling.infrastructure.factories.transformer import TransformerFactory
from ....shared.config import ROOT_PATH
from ....shared.infrastructure.processing.files.json import JsonFileHandler
from ....shared.infrastructure.processing.files.pickle import PickleFileHandler
from ....shared.infrastructure.processing.files.toml import TomlFileHandler
from ...domain.entities.inverse_mapping_engine import InverseMappingEngine
from ...domain.interfaces.base_inverse_mapping_engine_repository import (
    BaseInverseMappingEngineRepository,
)
from ...domain.value_objects.data_split import DataSplit
from ...domain.value_objects.transform_pipeline import TransformPipeline


class VersionManager:
    def __init__(self, toml_handler: TomlFileHandler):
        self._toml_handler = toml_handler

    def get_next_version(self, solver_directory: Path) -> int:
        existing_versions = []
        if solver_directory.exists():
            for entry in solver_directory.iterdir():
                if entry.is_dir():
                    try:
                        metadata_toml = entry / "metadata.toml"
                        if metadata_toml.exists():
                            payload = self._toml_handler.load(metadata_toml)
                            meta = payload.get("metadata", {})
                            vn = meta.get("version")
                            if isinstance(vn, int):
                                existing_versions.append(vn)
                    except Exception:
                        continue
        return max(existing_versions) + 1 if existing_versions else 1


class FileSystemInverseMappingEngineRepository(BaseInverseMappingEngineRepository):
    """
    File system implementation of BaseInverseMappingEngineRepository.
    Persists InverseMappingEngine as an opaque solver blob + structured transforms.
    Supports human-readable versioning: contexts/<dataset>/<solver_type>/v<N>-<timestamp>/
    """

    def __init__(self, transformer_factory: TransformerFactory | None = None):
        self._base_storage_path = ROOT_PATH / "contexts"
        self._toml_file_handler = TomlFileHandler()
        self._json_file_handler = JsonFileHandler()
        self._pickle_handler = PickleFileHandler()
        self._transformer_factory = transformer_factory or TransformerFactory()
        self._version_manager = VersionManager(self._toml_file_handler)
        self._base_storage_path.mkdir(parents=True, exist_ok=True)

    def _get_solver_dir(self, dataset_name: str, solver_type: str) -> Path:
        return self._base_storage_path / dataset_name / solver_type

    def _find_version_dir(self, solver_directory: Path, version: int) -> Path | None:
        if not solver_directory.exists():
            return None
        prefix = f"v{version}-"
        for entry in solver_directory.iterdir():
            if entry.is_dir() and entry.name.startswith(prefix):
                return entry
        return None

    def save(self, engine: InverseMappingEngine) -> None:
        """
        Persists an InverseMappingEngine entity.
        Layout: contexts/<dataset>/<solver_type>/v<N>-<timestamp>/
        """
        solver_type = engine.solver.type()
        solver_dir = self._get_solver_dir(engine.dataset_name, solver_type)
        solver_dir.mkdir(parents=True, exist_ok=True)

        next_version = self._version_manager.get_next_version(solver_dir)
        timestamp = engine.created_at.strftime("%Y-%m-%d_%H-%M-%S")
        version_name = f"v{next_version}-{timestamp}"
        engine_dir = solver_dir / version_name
        engine_dir.mkdir(exist_ok=True)

        # 1. Save metadata
        metadata = {
            "version": next_version,
            "dataset_name": engine.dataset_name,
            "solver_type": solver_type,
            "created_at": engine.created_at.isoformat(),
        }
        self._toml_file_handler.save(
            {"metadata": metadata}, engine_dir / "metadata.toml"
        )

        # 2. Pickle the entire solver blob (owns its mathematical/indices state)
        self._pickle_handler.save(engine.solver, engine_dir / "solver.pkl")

        # 3. Save transforms (structured for reconstruction via factory)
        t_base_dir = engine_dir / "transforms"
        t_base_dir.mkdir(exist_ok=True)
        for idx, (label, transform) in enumerate(engine.transform_pipeline.transforms):
            t_type = transform.config.get("type", "unknown")
            t_dir = t_base_dir / f"{idx}_{t_type}"
            t_dir.mkdir(exist_ok=True)

            config_to_save = transform.config.copy()
            config_to_save["target_label"] = label

            self._json_file_handler.save(config_to_save, t_dir / "config.json")
            self._pickle_handler.save(
                transform.get_fitted_state(), t_dir / "fitted_state.pkl"
            )

        # 4. Save DataSplit
        split_data = {
            "split_ratio": engine.data_split.split_ratio,
            "random_state": engine.data_split.random_state,
        }
        self._json_file_handler.save(split_data, engine_dir / "data_split.json")
        np.save(engine_dir / "train_indices.npy", engine.data_split.train_indices)
        np.save(engine_dir / "test_indices.npy", engine.data_split.test_indices)

    def load(
        self, dataset_name: str, solver_type: str, version: int | None = None
    ) -> InverseMappingEngine:
        """
        Loads an engine. If version is None, returns the latest one.
        """
        solver_dir = self._get_solver_dir(dataset_name, solver_type)
        if not solver_dir.exists():
            raise FileNotFoundError(
                f"No engines found for dataset '{dataset_name}' and solver '{solver_type}'"
            )

        if version is None:
            summaries = self.list(dataset_name, solver_type)
            if not summaries:
                raise FileNotFoundError(
                    f"No valid engines found for dataset '{dataset_name}' and solver '{solver_type}'"
                )
            version = summaries[0]["version"]

        version_dir = self._find_version_dir(solver_dir, version)
        if not version_dir or not version_dir.exists():
            raise FileNotFoundError(
                f"Engine version '{version}' for dataset '{dataset_name}' and solver '{solver_type}' not found."
            )

        return self._load_from_dir(version_dir)

    def get_version_by_number(
        self, dataset_name: str, solver_type: str, version: int
    ) -> InverseMappingEngine:
        return self.load(dataset_name, solver_type, version)

    def _load_from_dir(self, engine_dir: Path) -> InverseMappingEngine:
        metadata_path = engine_dir / "metadata.toml"
        solver_path = engine_dir / "solver.pkl"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found in '{engine_dir}'")
        if not solver_path.exists():
            raise FileNotFoundError(f"Solver state not found in '{engine_dir}'")

        payload = self._toml_file_handler.load(metadata_path)
        metadata = payload.get("metadata", {})

        # 1. Unpickle the solver
        solver = self._pickle_handler.load(solver_path)

        # 2. Load and reconstruct transforms
        transforms_list = []
        t_base_dir = engine_dir / "transforms"
        if t_base_dir.exists():
            subdirs = sorted(
                [d for d in t_base_dir.iterdir() if d.is_dir()],
                key=lambda x: int(x.name.split("_")[0]),
            )
            for d in subdirs:
                config = self._json_file_handler.load(d / "config.json")
                state = self._pickle_handler.load(d / "fitted_state.pkl")
                label = config.pop("target_label", "unknown")
                transform = TransformerFactory.from_checkpoint(config, state)
                transforms_list.append((label, transform))

        # 3. Load DataSplit
        split_path = engine_dir / "data_split.json"
        if split_path.exists():
            split_meta = self._json_file_handler.load(split_path)
        else:
            split_meta = {"split_ratio": 0.2, "random_state": 42}

        train_indices_path = engine_dir / "train_indices.npy"
        test_indices_path = engine_dir / "test_indices.npy"

        train_indices = (
            np.load(train_indices_path)
            if train_indices_path.exists()
            else np.array([], dtype=int)
        )
        test_indices = (
            np.load(test_indices_path)
            if test_indices_path.exists()
            else np.array([], dtype=int)
        )

        data_split = DataSplit(
            train_indices=train_indices,
            test_indices=test_indices,
            split_ratio=split_meta.get("split_ratio", 0.2),
            random_state=split_meta.get("random_state", 42),
        )

        return InverseMappingEngine(
            dataset_name=metadata["dataset_name"],
            solver=solver,
            transform_pipeline=TransformPipeline(transforms=transforms_list),
            data_split=data_split,
            created_at=datetime.fromisoformat(metadata["created_at"]),
        )

    def list(self, dataset_name: str, solver_type: str | None = None) -> list[dict]:
        """
        Lists engine versions. If solver_type is provided, lists for that solver.
        Otherwise, lists all for the dataset.
        """
        dataset_dir = self._base_storage_path / dataset_name
        if not dataset_dir.exists():
            return []

        engine_summaries = []

        solver_dirs = (
            [dataset_dir / solver_type]
            if solver_type
            else [d for d in dataset_dir.iterdir() if d.is_dir()]
        )

        for s_dir in solver_dirs:
            if not s_dir.exists():
                continue
            for v_dir in s_dir.iterdir():
                if v_dir.is_dir() and v_dir.name.startswith("v"):
                    metadata_path = v_dir / "metadata.toml"
                    if metadata_path.exists():
                        try:
                            payload = self._toml_file_handler.load(metadata_path)
                            engine_summaries.append(payload.get("metadata", {}))
                        except Exception:
                            continue

        return sorted(engine_summaries, key=lambda x: x["created_at"], reverse=True)
