from pathlib import Path
from typing import Any

from ....shared.config import ROOT_PATH
from ....shared.infrastructure.processing.files.json import JsonFileHandler
from ....shared.infrastructure.processing.files.pickle import PickleFileHandler
from ...domain.entities.dataset import Dataset
from ...domain.interfaces.base_repository import BaseDatasetRepository
from ...domain.value_objects.metadata import DatasetMetadata
from ...domain.value_objects.pareto import Pareto


class FileSystemDatasetRepository(BaseDatasetRepository):
    """
    Unified repository that persists the Dataset aggregate to the filesystem.
    It saves data under `data/<dataset_name>/raw`.
    """

    def __init__(self, file_path: str | Path = "data") -> None:
        self.base_path = ROOT_PATH / file_path
        self._pkl = PickleFileHandler()
        self._json = JsonFileHandler()

    def save(self, dataset: Dataset) -> Path:
        """Persist the Dataset aggregate."""
        self._save_raw(dataset)
        return self.base_path / dataset.name

    def delete(self, name: str) -> None:
        """Deletes a dataset and all its files from the filesystem."""
        dataset_dir = self.base_path / name
        if dataset_dir.exists():
            import shutil

            shutil.rmtree(dataset_dir)

    def load(self, name: str) -> Dataset:
        """Load the Dataset aggregate."""
        raw_payload = self._load_raw_payload(name)
        return self._rebuild_dataset(raw_payload, name)

    def _save_raw(self, dataset: Dataset) -> None:
        """Saves essential fields to the raw directory."""
        payload = {
            "name": dataset.name,
            "X": dataset.X,
            "y": dataset.y,
            "pareto": dataset.pareto,
            "metadata": dataset.metadata,
            "train_indices": dataset.train_indices,
            "test_indices": dataset.test_indices,
        }
        raw_dir = self._raw_dir(dataset.name)
        raw_dir.mkdir(parents=True, exist_ok=True)
        target = raw_dir / "dataset"
        self._pkl.save(payload, target)

    def _load_raw_payload(self, name: str) -> dict[str, Any]:
        """Loads the raw payload from the filesystem."""
        raw_dir = self._raw_dir(name)
        candidate = raw_dir / "dataset"

        if candidate.exists() or candidate.with_suffix(".pkl").exists():
            return self._pkl.load(candidate)

        raise FileNotFoundError(f"Dataset '{name}' not found at {candidate}")

    def _rebuild_dataset(self, payload: dict[str, Any], name: str) -> Dataset:
        """Reconstructs the Dataset aggregate from payload."""
        pareto_payload = payload.get("pareto")
        pareto = None
        if pareto_payload:
            if isinstance(pareto_payload, Pareto):
                pareto = pareto_payload
            else:
                pareto = Pareto(
                    set=pareto_payload.get("set"),
                    front=pareto_payload.get("front"),
                )

        # Core data (X, y)
        X = payload.get("X")
        y = payload.get("y")

        if X is None or y is None:
            raise ValueError(f"Dataset '{name}' is missing core data (X or y).")

        # Metadata extraction
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            metadata = DatasetMetadata(**metadata)
        elif metadata is None:
            # Fallback for old data or partial payloads
            metadata = DatasetMetadata(
                split_ratio=payload.get("split_ratio", 0.0),
                random_state=payload.get("random_state", 42),
            )

        # Indices
        train_indices = payload.get("train_indices")
        test_indices = payload.get("test_indices")

        if train_indices is None or test_indices is None:
            # Reconstruct indices if missing
            import numpy as np
            from sklearn.model_selection import train_test_split

            n_samples = len(X)
            indices = np.arange(n_samples)
            if metadata.split_ratio > 0.0:
                train_indices, test_indices = train_test_split(
                    indices,
                    test_size=metadata.split_ratio,
                    random_state=metadata.random_state,
                    shuffle=True,
                )
            else:
                train_indices = indices
                test_indices = np.array([], dtype=int)

        # Re-verify metadata has correct counts if they were missing or zero
        if metadata.n_samples == 0:
            metadata.n_samples = len(X)
            metadata.n_train = len(train_indices)
            metadata.n_test = len(test_indices)

        return Dataset.create(
            name=payload.get("name", name),
            X=X,
            y=y,
            metadata=metadata,
            train_indices=train_indices,
            test_indices=test_indices,
            pareto=pareto,
        )

    def list_all(self) -> list[str]:
        """List all available dataset names by checking for raw subdirectories."""
        datasets = []
        if not self.base_path.exists():
            return []

        for p in self.base_path.iterdir():
            if p.is_dir() and self._raw_dir(p.name).exists():
                datasets.append(p.name)
        return sorted(datasets)

    def _raw_dir(self, name: str) -> Path:
        """Returns the raw directory for the given dataset."""
        return self.base_path / name / "raw"
