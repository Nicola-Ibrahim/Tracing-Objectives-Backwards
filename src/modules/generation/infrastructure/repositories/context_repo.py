from pathlib import Path

import numpy as np

from ....modeling.infrastructure.factories.transformer import TransformerFactory
from ....shared.config import ROOT_PATH
from ....shared.infrastructure.processing.files.json import JsonFileHandler
from ....shared.infrastructure.processing.files.pickle import PickleFileHandler
from ....shared.infrastructure.processing.files.toml import TomlFileHandler
from ...domain.entities.generation_context import GenerationContext
from ...domain.interfaces.base_context_repository import BaseContextRepository


class FileSystemContextRepository(BaseContextRepository):
    """
    File system implementation of BaseContextRepository using TOML + npz + separate transformer folders.
    """

    def __init__(self, transformer_factory: TransformerFactory | None = None):
        self._base_storage_path = ROOT_PATH / "contexts"
        self._toml_file_handler = TomlFileHandler()
        self._json_file_handler = JsonFileHandler()
        self._pickle_handler = PickleFileHandler()
        self._transformer_factory = transformer_factory or TransformerFactory()
        self._base_storage_path.mkdir(parents=True, exist_ok=True)

    def _get_context_dir(self, dataset_name: str) -> Path:
        return self._base_storage_path / dataset_name

    def save(self, context: GenerationContext) -> None:
        context_dir = self._get_context_dir(context.dataset_name)
        context_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "dataset_name": context.dataset_name,
            "tau": context.tau,
            "is_trained": context.is_trained,
            "created_at": context.created_at.isoformat(),
        }

        metadata_path = context_dir / "metadata.toml"
        self._toml_file_handler.save({"metadata": metadata}, metadata_path)

        arrays_path = context_dir / "arrays.npz"
        np.savez(
            arrays_path,
            space_points=context.space_points,
            decision_vertices=context.decision_vertices,
        )

        def save_transforms(transforms_list, folder_name):
            t_base_dir = context_dir / folder_name
            t_base_dir.mkdir(exist_ok=True)
            for idx, transform_tuple in enumerate(transforms_list):
                transform, target = transform_tuple
                t_type = transform.config.get("type", "unknown")
                t_dir = t_base_dir / f"{idx}_{t_type}"
                t_dir.mkdir(exist_ok=True)

                # Copy config and append target for restoration later
                config_to_save = transform.config.copy()
                config_to_save["target"] = (
                    target.value if hasattr(target, "value") else str(target)
                )

                self._json_file_handler.save(config_to_save, t_dir / "config.json")
                self._pickle_handler.save(
                    transform.get_fitted_state(), t_dir / "fitted_state.pkl"
                )

        save_transforms(context.transforms, "transforms")

        surrogate_path = context_dir / "surrogate_step.pkl"
        self._pickle_handler.save(context.surrogate_estimator, surrogate_path)

        mesh_path = context_dir / "mesh.pkl"
        self._pickle_handler.save(context.mesh, mesh_path)

        knn_path = context_dir / "objective_knn.pkl"
        self._pickle_handler.save(context.objective_knn, knn_path)

    def load(self, dataset_name: str) -> GenerationContext:
        context_dir = self._get_context_dir(dataset_name)
        if not context_dir.exists():
            raise FileNotFoundError(
                f"GenerationContext for dataset '{dataset_name}' not found."
            )

        metadata_path = context_dir / "metadata.toml"
        arrays_path = context_dir / "arrays.npz"
        surrogate_path = context_dir / "surrogate_step.pkl"
        mesh_path = context_dir / "mesh.pkl"
        knn_path = context_dir / "objective_knn.pkl"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found for dataset '{dataset_name}'.")
        if not arrays_path.exists():
            raise FileNotFoundError(f"Arrays not found for dataset '{dataset_name}'.")
        if not surrogate_path.exists():
            raise FileNotFoundError(
                f"Surrogate model not found for dataset '{dataset_name}'."
            )
        if not mesh_path.exists():
            raise FileNotFoundError(
                f"Delaunay mesh not found for dataset '{dataset_name}'."
            )
        if not knn_path.exists():
            raise FileNotFoundError(
                f"Objective KNN index not found for dataset '{dataset_name}'."
            )

        payload = self._toml_file_handler.load(metadata_path)
        metadata = payload.get("metadata", {})

        with np.load(arrays_path) as arrays:
            space_points = arrays["space_points"]
            decision_vertices = arrays["decision_vertices"]

        def load_transforms(folder_name):
            transforms = []
            t_base_dir = context_dir / folder_name
            if t_base_dir.exists():
                subdirs = sorted(
                    [d for d in t_base_dir.iterdir() if d.is_dir()],
                    key=lambda x: int(x.name.split("_")[0]),
                )
                for d in subdirs:
                    config = self._json_file_handler.load(d / "config.json")
                    state = self._pickle_handler.load(d / "fitted_state.pkl")

                    # Extract target string if saved
                    target_str = config.pop("target", None)

                    transform = TransformerFactory.from_checkpoint(config, state)

                    if target_str:
                        from ....modeling.domain.interfaces.base_transform import (
                            TransformTarget,
                        )

                        try:
                            # Attempt enum mapping
                            target = TransformTarget(target_str)
                        except ValueError:
                            # Fallback to string literal
                            target = target_str
                        transforms.append((transform, target))
                    else:
                        transforms.append(transform)
            return transforms

        transforms = load_transforms("transforms")

        surrogate_estimator = self._pickle_handler.load(surrogate_path)
        mesh = self._pickle_handler.load(mesh_path)
        objective_knn = self._pickle_handler.load(knn_path)

        return GenerationContext(
            dataset_name=dataset_name,
            space_points=space_points,
            decision_vertices=decision_vertices,
            tau=metadata["tau"],
            transforms=transforms,
            surrogate_estimator=surrogate_estimator,
            objective_knn=objective_knn,
            mesh=mesh,
            is_trained=metadata.get("is_trained", True),
        )
