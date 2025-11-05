from pathlib import Path
from typing import Any, Literal, Union

from ....domain.datasets.entities.generated_dataset import GeneratedDataset
from ....domain.datasets.entities.processed_dataset import ProcessedDataset
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ....domain.datasets.value_objects.pareto import Pareto
from ...processing.files.json import JsonFileHandler
from ...processing.files.pickle import PickleFileHandler


class FileSystemDatasetRepository(BaseDatasetRepository):
    """
    Unified repository that persists both the raw generated dataset and its processed,
    normalized counterpart. Supports loading either representation on demand.
    """

    def __init__(self) -> None:
        super().__init__(file_path="data")
        self._raw_dir = self.base_path / "raw"
        self._processed_dir = self.base_path / "processed"
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._processed_dir.mkdir(parents=True, exist_ok=True)

        self._pkl = PickleFileHandler()
        self._json = JsonFileHandler()

    # ------------------------------------------------------------------
    # BaseDatasetRepository API
    # ------------------------------------------------------------------

    def save(
        self, *, raw: GeneratedDataset, processed: ProcessedDataset
    ) -> Path:
        """Persist raw and processed variants to disk."""
        self._save_raw(raw)
        self._save_processed(processed)
        return self.base_path

    def load(
        self, filename: str, variant: Literal["raw", "processed"] = "processed"
    ) -> Union[GeneratedDataset, ProcessedDataset]:
        if variant == "raw":
            return self._load_raw(filename)
        if variant == "processed":
            return self._load_processed(filename)
        raise ValueError(f"Unsupported dataset variant: {variant!r}")

    # ------------------------------------------------------------------
    # Raw dataset helpers
    # ------------------------------------------------------------------

    def _save_raw(self, data: GeneratedDataset) -> None:
        payload = data.model_dump()
        target = self._raw_dir / data.name
        self._pkl.save(payload, target)

    def _load_raw(self, filename: str) -> GeneratedDataset:
        stem = Path(filename).stem
        candidates = [
            self._raw_dir / stem,
            # Legacy layout: data/raw/<stem>.pkl
            self.base_path / stem,
        ]

        last_error: Exception | None = None
        for candidate in candidates:
            try:
                payload = self._pkl.load(candidate)
                return self._rebuild_generated_dataset(payload, fallback_name=stem)
            except FileNotFoundError as error:
                last_error = error
                continue

        if last_error:
            raise last_error
        raise FileNotFoundError(f"Raw dataset '{filename}' not found.")

    @staticmethod
    def _rebuild_generated_dataset(
        payload: dict[str, Any], fallback_name: str
    ) -> GeneratedDataset:
        pareto_payload = payload.get("pareto", {})
        pareto = (
            pareto_payload
            if isinstance(pareto_payload, Pareto)
            else Pareto(
                set=pareto_payload.get("set"),
                front=pareto_payload.get("front"),
            )
        )

        return GeneratedDataset.create(
            name=payload.get("name", fallback_name),
            X=payload.get("X"),
            y=payload.get("y"),
            pareto=pareto,
        )

    # ------------------------------------------------------------------
    # Processed dataset helpers
    # ------------------------------------------------------------------

    def _save_processed(self, data: ProcessedDataset) -> None:
        target_dir = self._processed_dir / data.name
        target_dir.mkdir(parents=True, exist_ok=True)

        dataset_pkl = target_dir / "dataset"
        x_norm_pkl = target_dir / "X_normalizer"
        y_norm_pkl = target_dir / "y_normalizer"
        meta_json = target_dir / "metadata.json"

        payload = data.model_dump(exclude={"X_normalizer", "y_normalizer"})
        self._pkl.save(payload, dataset_pkl)
        self._pkl.save(data.X_normalizer, x_norm_pkl)
        self._pkl.save(data.y_normalizer, y_norm_pkl)

        metadata: dict[str, Any] = {
            "shapes": {
                "X_train": list(data.X_train.shape),
                "y_train": list(data.y_train.shape),
                "X_test": list(data.X_test.shape),
                "y_test": list(data.y_test.shape),
                "pareto_set": list(data.pareto.set.shape),
                "pareto_front": list(data.pareto.front.shape),
            },
            "metadata": data.metadata,
        }
        self._json.save(metadata, meta_json)

    def _load_processed(self, filename: str) -> ProcessedDataset:
        stem = Path(filename).stem
        modern_dir = self._processed_dir / stem
        if modern_dir.exists():
            return self._load_processed_from_directory(modern_dir, fallback_name=stem)

        # Backwards compatibility with legacy layout (<data/processed>/dataset.pkl)
        legacy_dir = self._processed_dir
        if legacy_dir.exists():
            try:
                return self._load_processed_from_directory(
                    legacy_dir, fallback_name=stem
                )
            except FileNotFoundError:
                pass

        # Legacy single-file fallback: <data/processed>/<stem>.pkl
        legacy_file = legacy_dir / stem
        try:
            payload = self._pkl.load(legacy_file)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Processed dataset '{stem}' not found. "
                "Ensure the dataset has been generated and processed."
            ) from error
        return self._rebuild_processed_dataset(
            payload=payload,
            fallback_name=payload.get("name") or stem,
            normalizers=(
                payload.get("X_normalizer"),
                payload.get("y_normalizer"),
            ),
            normalizer_files=(
                legacy_dir / f"{stem}_X_normalizer",
                legacy_dir / f"{stem}_y_normalizer",
            ),
        )

    def _load_processed_from_directory(
        self, directory: Path, fallback_name: str
    ) -> ProcessedDataset:
        dataset_pkl = directory / "dataset"
        x_norm_pkl = directory / "X_normalizer"
        y_norm_pkl = directory / "y_normalizer"

        if not dataset_pkl.with_suffix(".pkl").exists():
            raise FileNotFoundError(f"Missing dataset payload: {dataset_pkl.with_suffix('.pkl')}")
        if not x_norm_pkl.with_suffix(".pkl").exists() or not y_norm_pkl.with_suffix(
            ".pkl"
        ).exists():
            raise FileNotFoundError(
                f"Missing normalizers in {directory}. Expected 'X_normalizer.pkl' and 'y_normalizer.pkl'."
            )

        payload = self._pkl.load(dataset_pkl)
        X_normalizer = self._pkl.load(x_norm_pkl)
        y_normalizer = self._pkl.load(y_norm_pkl)

        return self._rebuild_processed_dataset(
            payload=payload,
            fallback_name=payload.get("name", fallback_name),
            normalizers=(X_normalizer, y_normalizer),
        )

    def _rebuild_processed_dataset(
        self,
        *,
        payload: dict[str, Any],
        fallback_name: str,
        normalizers: tuple[Any | None, Any | None],
        normalizer_files: tuple[Path, Path] | None = None,
    ) -> ProcessedDataset:
        X_normalizer, y_normalizer = normalizers

        if (X_normalizer is None or y_normalizer is None) and normalizer_files:
            sib_x, sib_y = normalizer_files
            if sib_x.with_suffix(".pkl").exists() and sib_y.with_suffix(".pkl").exists():
                X_normalizer = self._pkl.load(sib_x)
                y_normalizer = self._pkl.load(sib_y)
            else:
                raise FileNotFoundError(
                    f"Legacy dataset missing embedded normalizers and no sibling files found for '{fallback_name}'."
                )

        pareto_payload = payload.get("pareto") or {
            "set": payload.get("pareto_set"),
            "front": payload.get("pareto_front"),
        }
        pareto = (
            pareto_payload
            if isinstance(pareto_payload, Pareto)
            else Pareto(
                set=pareto_payload.get("set"),
                front=pareto_payload.get("front"),
            )
        )

        return ProcessedDataset.create(
            name=fallback_name,
            X_train=payload.get("X_train"),
            y_train=payload.get("y_train"),
            X_test=payload.get("X_test"),
            y_test=payload.get("y_test"),
            X_normalizer=X_normalizer,
            y_normalizer=y_normalizer,
            pareto=pareto,
            metadata=payload.get("metadata"),
        )
