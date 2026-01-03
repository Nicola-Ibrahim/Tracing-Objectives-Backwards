from pathlib import Path
from typing import Any

from ....domain.datasets.entities.dataset import Dataset
from ....domain.datasets.entities.processed_data import ProcessedData
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ....domain.datasets.value_objects.pareto import Pareto
from ...processing.files.json import JsonFileHandler
from ...processing.files.pickle import PickleFileHandler


class FileSystemDatasetRepository(BaseDatasetRepository):
    """
    Unified repository that persists the Dataset aggregate.
    It saves data under `data/<dataset_name>/{raw,processed}` with legacy fallbacks.
    """

    def __init__(self, file_path: str | Path = "data") -> None:
        super().__init__(file_path=file_path)
        self._legacy_raw_dir = self.base_path / "raw"
        self._legacy_processed_dir = self.base_path / "processed"

        self._pkl = PickleFileHandler()
        self._json = JsonFileHandler()

    def save(self, dataset: Dataset) -> Path:
        """Persist the Dataset aggregate."""
        # Save raw part
        self._save_raw(dataset)

        # Save processed part if it exists
        if dataset.processed:
            self._save_processed(dataset)

        return self.base_path / dataset.name

    def load(self, name: str) -> Dataset:
        """
        Load the Dataset aggregate.
        Always loads the raw data.
        If processed data exists for this name, it loads that as well and attaches it.
        """
        # Load raw data
        raw_payload = self._load_raw_payload(name)
        dataset = self._rebuild_dataset(raw_payload, name)

        # Try to load processed data
        try:
            processed = self._load_processed_part(name)
            dataset.processed = processed
        except FileNotFoundError:
            # It's fine if processed data doesn't exist yet
            pass

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

    # ------------------------------------------------------------------
    # Processed part helpers
    # ------------------------------------------------------------------

    def _save_processed(self, dataset: Dataset) -> None:
        processed = dataset.processed
        target_dir = self._processed_dir(dataset.name)
        target_dir.mkdir(parents=True, exist_ok=True)

        dataset_pkl = target_dir / "dataset"
        x_norm_pkl = target_dir / "decisions_normalizer"
        y_norm_pkl = target_dir / "objectives_normalizer"
        meta_json = target_dir / "metadata.json"

        # Dump processed fields
        payload = {
            "decisions_train": processed.decisions_train,
            "objectives_train": processed.objectives_train,
            "decisions_test": processed.decisions_test,
            "objectives_test": processed.objectives_test,
            "metadata": processed.metadata,
        }

        self._pkl.save(payload, dataset_pkl)
        self._pkl.save(processed.decisions_normalizer, x_norm_pkl)
        self._pkl.save(processed.objectives_normalizer, y_norm_pkl)

        metadata_summary: dict[str, Any] = {
            "shapes": {
                "decisions_train": list(processed.decisions_train.shape),
                "objectives_train": list(processed.objectives_train.shape),
                "decisions_test": list(processed.decisions_test.shape),
                "objectives_test": list(processed.objectives_test.shape),
            },
            "metadata": processed.metadata,
        }
        self._json.save(metadata_summary, meta_json)

    def _load_processed_part(self, name: str) -> ProcessedData:
        directory = self._processed_dir(name)
        if not directory.exists():
            raise FileNotFoundError(
                f"Processed data directory for '{name}' not found."
            )

        dataset_pkl = directory / "dataset"
        decisions_norm_pkl = directory / "decisions_normalizer"
        objectives_norm_pkl = directory / "objectives_normalizer"

        if not dataset_pkl.with_suffix(".pkl").exists():
            # Fallback logic for legacy structures if needed, or fail
            raise FileNotFoundError(f"Missing processed dataset payload for '{name}'")

        payload = self._pkl.load(dataset_pkl)
        decisions_normalizer = self._pkl.load(decisions_norm_pkl)
        objectives_normalizer = self._pkl.load(objectives_norm_pkl)

        return ProcessedData.create(
            decisions_train=payload.get("decisions_train"),
            objectives_train=payload.get("objectives_train"),
            decisions_test=payload.get("decisions_test"),
            objectives_test=payload.get("objectives_test"),
            decisions_normalizer=decisions_normalizer,
            objectives_normalizer=objectives_normalizer,
            metadata=payload.get("metadata"),
        )

    def _raw_dir(self, name: str) -> Path:
        return self.base_path / name / "raw"

    def _processed_dir(self, name: str) -> Path:
        return self.base_path / name / "processed"
