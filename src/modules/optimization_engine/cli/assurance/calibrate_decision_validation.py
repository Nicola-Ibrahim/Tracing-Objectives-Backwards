from typing import Type

import click

from ...application.assurance.decision_validation.command import (
    CalibrateDecisionValidationCommand,
    ConformalCalibratorParams,
    OODCalibratorParams,
)
from ...application.assurance.decision_validation.handler import (
    CalibrateDecisionValidationCommandHandler,
)
from ...application.dtos import (
    COCOEstimatorParams,
    CVAEEstimatorParams,
    CVAEMDNEstimatorParams,
    EstimatorParams,
    GaussianProcessEstimatorParams,
    MDNEstimatorParams,
    NeuralNetworkEstimatorParams,
    RBFEstimatorParams,
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

INVERSE_ESTIMATOR_REGISTRY: dict[str, Type[EstimatorParams]] = {
    "cvae": CVAEEstimatorParams,
    "cvae_mdn": CVAEMDNEstimatorParams,
    "gaussian_process": GaussianProcessEstimatorParams,
    "mdn": MDNEstimatorParams,
    "neural_network": NeuralNetworkEstimatorParams,
    "rbf": RBFEstimatorParams,
    "coco": COCOEstimatorParams,
}


def _create_estimator_params(
    estimator_key: str,
) -> EstimatorParams:
    try:
        params_cls = INVERSE_ESTIMATOR_REGISTRY[estimator_key]
    except KeyError as exc:
        raise ValueError(f"Unsupported estimator '{estimator_key}'") from exc
    return params_cls()


@click.command(help="Calibrate decision-validation gates for a trained estimator")
@click.option(
    "--estimator",
    type=click.Choice(sorted(INVERSE_ESTIMATOR_REGISTRY.keys())),
    default="coco",
    show_default=True,
    help="Estimator configuration used for calibration",
)
def cli(estimator: str) -> None:
    estimator_params_model = _create_estimator_params(estimator)

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
