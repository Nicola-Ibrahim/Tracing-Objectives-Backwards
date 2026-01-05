from datetime import datetime

from .....shared.config import ROOT_PATH
from ....domain.assurance.decision_validation.entities.decision_validation_calibration import (
    DecisionValidationCalibration,
)
from ....domain.assurance.decision_validation.interfaces import (
    BaseDecisionValidationCalibrationRepository,
)
from ...processing.files.json import JsonFileHandler
from ...processing.files.pickle import PickleFileHandler


class FileSystemDecisionValidationCalibrationRepository(
    BaseDecisionValidationCalibrationRepository
):
    """Persist calibrations under models/calibrate."""

    def __init__(self) -> None:
        self._base_path = ROOT_PATH / "models" / "calibrate"
        self._json_file_handler = JsonFileHandler()
        self._pickle_file_handler = PickleFileHandler()
        self._base_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, calibration: DecisionValidationCalibration) -> None:
        self._base_path.mkdir(parents=True, exist_ok=True)

        self._pickle_file_handler.save(
            calibration.ood_validator, self._base_path / "ood_validator.pkl"
        )
        self._pickle_file_handler.save(
            calibration.conformal_validator,
            self._base_path / "conformal_validator.pkl",
        )

        metadata = {
            "id": calibration.id,
            "created_at": calibration.created_at.isoformat(),
            "ood_validator": calibration.ood_validator.describe(),
            "conformal_validator": calibration.conformal_validator.describe(),
        }
        self._json_file_handler.save(metadata, self._base_path / "metadata.json")

    def load(self) -> DecisionValidationCalibration:
        metadata_path = self._base_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError("No stored decision validation calibration found.")

        payload = self._json_file_handler.load(metadata_path)
        created_at_raw = payload.get("created_at")
        if isinstance(created_at_raw, str):
            created_at = datetime.fromisoformat(created_at_raw)
        elif isinstance(created_at_raw, datetime):
            created_at = created_at_raw
        else:
            raise ValueError("Calibration metadata missing 'created_at'.")

        ood_path = self._base_path / "ood_validator.pkl"
        conformal_path = self._base_path / "conformal_validator.pkl"
        if not ood_path.exists():
            ood_path = self._base_path / "ood_calibrator.pkl"
        if not conformal_path.exists():
            conformal_path = self._base_path / "conformal_calibrator.pkl"
        if not ood_path.exists() or not conformal_path.exists():
            raise FileNotFoundError(
                "Stored calibration is incomplete: missing validator artifacts."
            )

        ood = self._pickle_file_handler.load(ood_path)
        conformal = self._pickle_file_handler.load(conformal_path)

        calibration_id = payload.get("id")
        if not isinstance(calibration_id, str):
            raise ValueError("Calibration metadata missing 'id'.")

        return DecisionValidationCalibration.from_data(
            id=calibration_id,
            created_at=created_at,
            ood_validator=ood,
            conformal_validator=conformal,
        )
