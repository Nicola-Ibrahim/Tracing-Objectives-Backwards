import click

from ...application.assurance.decision_validation.calibrate_decision_validation_command import (
    CalibrateDecisionValidationCommand,
    ConformalCalibratorParams,
    OODCalibratorParams,
)
from ...application.assurance.decision_validation.calibrate_decision_validation_handler import (
    CalibrateDecisionValidationCommandHandler,
)
from ...application.factories.assurance import (
    ConformalCalibratorFactory,
    OODCalibratorFactory,
)
from ...application.factories.estimator import EstimatorFactory
from ...infrastructure.assurance.repositories.calibration_repository import (
    FileSystemDecisionValidationCalibrationRepository,
)
from ...infrastructure.datasets.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...infrastructure.loggers.cmd_logger import CMDLogger
from ..common import INVERSE_ESTIMATOR_REGISTRY, create_estimator_params


@click.command(help="Calibrate decision-validation gates for a trained estimator")
@click.option(
    "--estimator",
    type=click.Choice(sorted(INVERSE_ESTIMATOR_REGISTRY.keys())),
    default="coco",
    show_default=True,
    help="Estimator configuration used for calibration",
)
def cli(estimator: str) -> None:
    estimator_params_model = create_estimator_params(
        estimator, registry=INVERSE_ESTIMATOR_REGISTRY
    )

    command = CalibrateDecisionValidationCommand(
        estimator_params=estimator_params_model,
        ood_calibrator_params=OODCalibratorParams(),
        conformal_calibrator_params=ConformalCalibratorParams(),
    )

    handler = CalibrateDecisionValidationCommandHandler(
        processed_data_repository=FileSystemDatasetRepository(),
        calibration_repository=FileSystemDecisionValidationCalibrationRepository(),
        estimator_factory=EstimatorFactory(),
        ood_calibrator_factory=OODCalibratorFactory(),
        conformal_calibrator_factory=ConformalCalibratorFactory(),
        logger=CMDLogger(name="AssuranceCalibrationLogger"),
    )

    handler.execute(command)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
