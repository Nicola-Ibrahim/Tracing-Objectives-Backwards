"""CLI entry point to fit and persist decision-validation calibrations."""

from __future__ import annotations

from ...application.assurance.decision_validation import (
    CalibrateDecisionValidationCommand,
    CalibrateDecisionValidationCommandHandler,
)
from ...application.factories.assurance import (
    create_conformal_calibrator,
    create_forward_model,
    create_ood_calibrator,
)
from ...application.factories.estimator import EstimatorFactory
from ...infrastructure.loggers.cmd_logger import CMDLogger
from ...infrastructure.repositories.assurance import (
    FileSystemDecisionValidationCalibrationRepository,
)
from ...infrastructure.repositories.datasets.processed_dataset_repo import (
    FileSystemProcessedDatasetRepository,
)


def main() -> None:
    command = CalibrateDecisionValidationCommand(
        scope="mdn",
        forward_model_configs=[{"type": "coco_biobj", "function_indices": 5}],
    )

    handler = CalibrateDecisionValidationCommandHandler(
        processed_data_repository=FileSystemProcessedDatasetRepository(),
        calibration_repository=FileSystemDecisionValidationCalibrationRepository(),
        forward_model_factory=EstimatorFactory().create_forward,
        forward_adapter_factory=create_forward_model,
        ood_calibrator_factory=lambda percentile, cov_reg: create_ood_calibrator(
            percentile=percentile, cov_reg=cov_reg
        ),
        conformal_calibrator_factory=lambda confidence: create_conformal_calibrator(
            confidence=confidence
        ),
        logger=CMDLogger(name="AssuranceCalibrationLogger"),
    )

    handler.execute(command)


if __name__ == "__main__":
    main()
