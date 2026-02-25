from pathlib import Path

import numpy as np

from ....shared.config import ROOT_PATH
from ....shared.infrastructure.processing.files.toml import TomlFileHandler
from ...domain.entities.coherence_context import CoherenceContext
from ...domain.interfaces.base_context_repository import BaseContextRepository


class FileSystemContextRepository(BaseContextRepository):
    """
    File system implementation of BaseContextRepository using TOML + npz.
    """

    def __init__(self):
        self._base_storage_path = ROOT_PATH / "contexts"
        self._toml_file_handler = TomlFileHandler()
        self._base_storage_path.mkdir(parents=True, exist_ok=True)

    def _get_context_dir(self, dataset_name: str) -> Path:
        return self._base_storage_path / dataset_name

    def save(self, context: CoherenceContext) -> None:
        context_dir = self._get_context_dir(context.dataset_name)
        context_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "dataset_name": context.dataset_name,
            "tau": context.tau,
            "k_neighbors": context.k_neighbors,
            "surrogate_type": context.surrogate_type,
            "surrogate_version": context.surrogate_version,
            "created_at": context.created_at.isoformat(),
        }

        metadata_path = context_dir / "metadata.toml"
        self._toml_file_handler.save({"metadata": metadata}, metadata_path)

        arrays_path = context_dir / "arrays.npz"
        np.savez(
            arrays_path,
            objectives=context.objectives,
            anchors_norm=context.anchors_norm,
        )

    def load(self, dataset_name: str) -> CoherenceContext:
        context_dir = self._get_context_dir(dataset_name)
        if not context_dir.exists():
            raise FileNotFoundError(
                f"CoherenceContext for dataset '{dataset_name}' not found."
            )

        metadata_path = context_dir / "metadata.toml"
        arrays_path = context_dir / "arrays.npz"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found for dataset '{dataset_name}'.")
        if not arrays_path.exists():
            raise FileNotFoundError(f"Arrays not found for dataset '{dataset_name}'.")

        payload = self._toml_file_handler.load(metadata_path)
        metadata = payload.get("metadata", {})

        with np.load(arrays_path) as arrays:
            objectives = arrays["objectives"]
            anchors_norm = arrays["anchors_norm"]

        return CoherenceContext(
            dataset_name=metadata["dataset_name"],
            objectives=objectives,
            anchors_norm=anchors_norm,
            tau=metadata["tau"],
            k_neighbors=metadata.get("k_neighbors", 5),
            surrogate_type=metadata["surrogate_type"],
            surrogate_version=metadata["surrogate_version"],
            created_at=metadata["created_at"],
        )
