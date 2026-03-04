from pathlib import Path
from typing import Any

from ....shared.config import ROOT_PATH
from ....shared.infrastructure.processing.files.json import JsonFileHandler
from ....shared.infrastructure.processing.files.pickle import PickleFileHandler
from ...domain.entities.dataset import Dataset
from ...domain.interfaces.base_repository import BaseDatasetRepository
from ...domain.value_objects.pareto import Pareto


class FileSystemDatasetRepository(BaseDatasetRepository):
    """
    Unified repository that persists the Dataset aggregate to the filesystem.
    It saves data under `data/<dataset_name>/raw`.
    """

    def __init__(self, file_path: str | Path = "data") -> None:
        self.base_path = ROOT_PATH / file_path
        self._legacy_raw_dir = self.base_path / "raw"

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
        """
        Load the Dataset aggregate.
        Always loads the raw data.
        """
        raw_payload = self._load_raw_payload(name)
        dataset = self._rebuild_dataset(raw_payload, name)
        return dataset

    # ------------------------------------------------------------------
    # Raw dataset helpers
    # ------------------------------------------------------------------

    def _save_raw(self, dataset: Dataset) -> None:
        # Dump only the raw fields
        payload = {
            "name": dataset.name,
            "decisions": dataset.decisions,
            "objectives": dataset.objectives,
            "pareto": dataset.pareto,
            "created_at": dataset.created_at,
        }
        raw_dir = self._raw_dir(dataset.name)
        raw_dir.mkdir(parents=True, exist_ok=True)
        target = raw_dir / "dataset"
        self._pkl.save(payload, target)

    def _load_raw_payload(self, name: str) -> dict[str, Any]:
        raw_dir = self._raw_dir(name)
        candidates = [
            raw_dir / "dataset",
            raw_dir / name,
        ]

        for candidate in candidates:
            if candidate.exists() or candidate.with_suffix(".pkl").exists():
                return self._pkl.load(candidate)

        raise FileNotFoundError(f"Raw dataset '{name}' not found.")

    def _rebuild_dataset(self, payload: dict[str, Any], name: str) -> Dataset:
        pareto_payload = payload.get("pareto", {})
        pareto = (
            pareto_payload
            if isinstance(pareto_payload, Pareto)
            else Pareto(
                set=pareto_payload.get("set"),
                front=pareto_payload.get("front"),
            )
        )

        return Dataset.create(
            name=payload.get("name", name),
            decisions=payload.get("decisions"),
            objectives=payload.get("objectives"),
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
        return self.base_path / name / "raw"
