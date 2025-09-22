from datetime import datetime
from pathlib import Path
from typing import Any

from .....shared.config import ROOT_PATH
from ....domain.datasets.entities.processed_dataset import ProcessedDataset
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ...processing.files.json import JsonFileHandler
from ...processing.files.pickle import PickleFileHandler


class FileSystemProcessedDatasetRepository(BaseDatasetRepository):
    """
    File-system repository for ProcessedDataset bundles.

    New layout (directory-based):
      <ROOT>/data/processed/<name>/
        ├─ dataset.pkl            # ProcessedDataset payload EXCLUDING normalizers
        ├─ X_normalizer.pkl       # fitted X normalizer (pickle)
        ├─ y_normalizer.pkl       # fitted y normalizer (pickle)
        └─ metadata.json          # small JSON with shapes, created_at, etc.

    Backward compatibility:
      If <ROOT>/data/processed/<name>.pkl exists, we load it (legacy single-file),
      and try to read embedded normalizers (if present). If not embedded, we
      optionally look for sibling normalizer files named:
        <ROOT>/data/processed/<name>_X_normalizer.pkl, <name>_y_normalizer.pkl
    """

    def __init__(self) -> None:
        super().__init__()
        self.base_path = ROOT_PATH / "data" / "processed"
        self.base_path.mkdir(parents=True, exist_ok=True)

        self._pkl = PickleFileHandler()
        self._json = JsonFileHandler()

    # -------- BaseDatasetRepository API --------

    def save(self, data: ProcessedDataset) -> Path:
        """
        Persist a processed dataset under <base>/<name>/.
        Returns the folder path.
        """

        dataset_pkl = self.base_path / "dataset.pkl"
        x_norm_pkl = self.base_path / "X_normalizer.pkl"
        y_norm_pkl = self.base_path / "y_normalizer.pkl"
        meta_json = self.base_path / "metadata.json"

        # 1) Save payload EXCLUDING normalizers to avoid duplication
        payload = data.model_dump(exclude={"X_normalizer", "y_normalizer"})
        self._pkl.save(payload, dataset_pkl)

        # 2) Save normalizers separately (mirrors FileSystemModelArtifactRepository)
        self._pkl.save(data.X_normalizer, x_norm_pkl)
        self._pkl.save(data.y_normalizer, y_norm_pkl)

        # 3) Save lightweight metadata
        metadata: dict[str, Any] = {
            "name": data.name,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "shapes": {
                "X_train": list(data.X_train.shape),
                "y_train": list(data.y_train.shape),
                "X_test": list(data.X_test.shape),
                "y_test": list(data.y_test.shape),
                "pareto_set": list(data.pareto_set.shape),
                "pareto_front": list(data.pareto_front.shape),
            },
            "has_normalizers": True,
        }
        self._json.save(metadata, meta_json)

        return self.base_path

    def load(self, filename: str) -> ProcessedDataset:
        """
        Load a processed dataset. Tries the new directory layout first, then legacy file.
        Accepts 'name' or 'name.pkl'.
        """
        # Prefer directory layout
        if self.base_path.exists() and self.base_path.is_dir():
            dataset_pkl = self.base_path / "dataset.pkl"
            x_norm_pkl = self.base_path / "X_normalizer.pkl"
            y_norm_pkl = self.base_path / "y_normalizer.pkl"

            if not dataset_pkl.exists():
                raise FileNotFoundError(f"Missing dataset payload: {dataset_pkl}")
            if not x_norm_pkl.exists() or not y_norm_pkl.exists():
                raise FileNotFoundError(
                    f"Missing normalizers in {self.base_path}. Expected 'X_normalizer.pkl' and 'y_normalizer.pkl'."
                )

            payload = self._pkl.load(dataset_pkl)
            X_normalizer = self._pkl.load(x_norm_pkl)
            y_normalizer = self._pkl.load(y_norm_pkl)

            name = payload.get("name") or self.base_path.name

            return ProcessedDataset.create(
                name=name,
                X_train=payload["X_train"],
                y_train=payload["y_train"],
                X_test=payload["X_test"],
                y_test=payload["y_test"],
                X_normalizer=X_normalizer,
                y_normalizer=y_normalizer,
                pareto_set=payload.get("pareto_set"),
                pareto_front=payload.get("pareto_front"),
                metadata=payload.get("metadata"),
            )

        # Legacy single-file fallback: <base>/<name>.pkl
        legacy_file = self._file_for(filename)
        if legacy_file.exists():
            payload = self._pkl.load(legacy_file)
            name = payload.get("name") or legacy_file.stem

            # Try embedded normalizers first (legacy format may have them)
            X_norm = payload.get("X_normalizer")
            y_norm = payload.get("y_normalizer")

            # If not embedded, try sibling files with suffixes
            if X_norm is None or y_norm is None:
                sib_X = self.base_path / f"{legacy_file.stem}_X_normalizer.pkl"
                sib_y = self.base_path / f"{legacy_file.stem}_y_normalizer.pkl"
                if sib_X.exists() and sib_y.exists():
                    X_norm = self._pkl.load(sib_X)
                    y_norm = self._pkl.load(sib_y)
                else:
                    raise FileNotFoundError(
                        "Legacy dataset missing embedded normalizers and no sibling "
                        f"normalizer files found for stem '{legacy_file.stem}'."
                    )

            return ProcessedDataset.create(
                name=name,
                X_train=payload["X_train"],
                y_train=payload["y_train"],
                X_test=payload["X_test"],
                y_test=payload["y_test"],
                X_normalizer=X_norm,
                y_normalizer=y_norm,
                pareto_set=payload.get("pareto_set"),
                pareto_front=payload.get("pareto_front"),
                metadata=payload.get("metadata"),
            )

        # Nothing found
        raise FileNotFoundError(
            f"Processed dataset not found as dir or file: "
            f"{self.base_path} or {legacy_file}"
        )

    # ----------------- helpers -----------------

    def _dir_for(self, name_or_filename: str) -> Path:
        stem = Path(name_or_filename).stem
        return self.base_path / stem

    def _file_for(self, name_or_filename: str) -> Path:
        stem = Path(name_or_filename).stem
        return self.base_path / f"{stem}.pkl"
