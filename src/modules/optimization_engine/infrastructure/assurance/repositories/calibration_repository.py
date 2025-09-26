from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .....shared.config import ROOT_PATH
from ....domain.assurance.decision_validation.entities import (
    DecisionValidationCalibration,
)
from ....domain.assurance.decision_validation.interfaces import (
    DecisionValidationCalibrationRepository,
)
from ...processing.files.json import JsonFileHandler
from ...processing.files.pickle import PickleFileHandler


@dataclass
class _CalibrationMetadata:
    directory: Path
    payload: dict

    @property
    def created_at(self) -> datetime:
        raw = self.payload.get("created_at")
        if isinstance(raw, str):
            return datetime.fromisoformat(raw)
        if isinstance(raw, datetime):
            return raw
        raise ValueError("Missing 'created_at' in calibration metadata")

    @property
    def id(self) -> str:
        value = self.payload.get("id")
        if not isinstance(value, str):
            raise ValueError("Calibration metadata missing 'id'")
        return value

    @property
    def version(self) -> int | None:
        value = self.payload.get("version")
        return int(value) if value is not None else None


class FileSystemDecisionValidationCalibrationRepository(
    DecisionValidationCalibrationRepository
):
    """Persist calibrations under models/assurance/decision_validation."""

    def __init__(self) -> None:
        self._base_path = ROOT_PATH / "models" / "assurance" / "decision_validation"
        self._json = JsonFileHandler()
        self._pickle = PickleFileHandler()
        self._base_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, calibration: DecisionValidationCalibration) -> None:
        scope_dir = self._base_path / calibration.scope
        scope_dir.mkdir(parents=True, exist_ok=True)

        version = self._next_version(scope_dir)
        timestamp = calibration.created_at.strftime("%Y-%m-%d_%H-%M-%S")
        directory = scope_dir / f"v{version}-{calibration.id}-{timestamp}"
        directory.mkdir(exist_ok=True)

        self._pickle.save(calibration.ood_calibrator, directory / "ood_calibrator.pkl")
        self._pickle.save(
            calibration.conformal_calibrator,
            directory / "conformal_calibrator.pkl",
        )

        metadata = {
            "id": calibration.id,
            "scope": calibration.scope,
            "created_at": calibration.created_at.isoformat(),
            "version": version,
        }
        self._json.save(metadata, directory / "metadata.json")
        calibration.version = version

    def load_latest(self, scope: str) -> DecisionValidationCalibration:
        scope_dir = self._base_path / scope
        candidates = list(self._iter_metadata(scope_dir))
        if not candidates:
            raise FileNotFoundError(f"No calibration found for scope '{scope}'.")
        candidates.sort(key=lambda item: item.created_at, reverse=True)
        return self._load_from_metadata(candidates[0])

    def load(self, scope: str, calibration_id: str) -> DecisionValidationCalibration:
        scope_dir = self._base_path / scope
        for meta in self._iter_metadata(scope_dir):
            if meta.id == calibration_id:
                return self._load_from_metadata(meta)
        raise FileNotFoundError(
            f"Calibration '{calibration_id}' not found for scope '{scope}'."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _iter_metadata(self, scope_dir: Path) -> Iterator[_CalibrationMetadata]:
        if not scope_dir.exists():
            return iter(())
        for entry in scope_dir.iterdir():
            if not entry.is_dir():
                continue
            metadata_path = entry / "metadata.json"
            if not metadata_path.exists():
                continue
            try:
                payload = self._json.load(metadata_path)
                yield _CalibrationMetadata(directory=entry, payload=payload)
            except Exception:
                continue

    def _load_from_metadata(
        self, metadata: _CalibrationMetadata
    ) -> DecisionValidationCalibration:
        directory = metadata.directory
        ood = self._pickle.load(directory / "ood_calibrator.pkl")
        conformal = self._pickle.load(directory / "conformal_calibrator.pkl")
        payload = metadata.payload
        created_at = payload["created_at"]
        if isinstance(created_at, str):
            created_at_dt = datetime.fromisoformat(created_at)
        else:
            created_at_dt = created_at

        return DecisionValidationCalibration.from_data(
            id=metadata.id,
            scope=payload.get("scope", "default"),
            created_at=created_at_dt,
            ood_calibrator=ood,
            conformal_calibrator=conformal,
            version=metadata.version,
        )

    @staticmethod
    def _next_version(scope_dir: Path) -> int:
        versions: list[int] = []
        if scope_dir.exists():
            for entry in scope_dir.iterdir():
                if not entry.is_dir():
                    continue
                parts = entry.name.split("-")
                if not parts or not parts[0].startswith("v"):
                    continue
                try:
                    versions.append(int(parts[0][1:]))
                except ValueError:
                    continue
        return max(versions) + 1 if versions else 1
