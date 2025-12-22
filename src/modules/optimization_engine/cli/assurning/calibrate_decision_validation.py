from typing import Type

import click

from ...application.assuring.calibrate_decision_validation.command import (
    CalibrateDecisionValidationCommand,
    ConformalValidatorParams,
    OODValidatorParams,
)
from ...application.assuring.calibrate_decision_validation.handler import (
    CalibrateDecisionValidationCommandHandler,
)
from ...application.dtos import (
    COCOEstimatorParams,
    CVAEEstimatorParams,
    EstimatorParams,
    GaussianProcessEstimatorParams,
    MDNEstimatorParams,
    NeuralNetworkEstimatorParams,
    RBFEstimatorParams,
)
from ...application.factories.assurance import (
    ConformalValidatorFactory,
    OODValidatorFactory,
)
from ...application.factories.estimator import EstimatorFactory
from ...infrastructure.assurance.repositories.calibration_repository import (
    FileSystemDecisionValidationCalibrationRepository,
)
from ...infrastructure.datasets.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...infrastructure.shared.loggers.cmd_logger import CMDLogger

INVERSE_ESTIMATOR_REGISTRY: dict[str, Type[EstimatorParams]] = {
    "cvae": CVAEEstimatorParams,
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
@click.option(
    "--dataset-name",
    default="dataset",
    show_default=True,
    help="Dataset identifier to load for calibration.",
)
def cli(estimator: str, dataset_name: str) -> None:
    estimator_params_model = _create_estimator_params(estimator)

    command = CalibrateDecisionValidationCommand(
        dataset_name=dataset_name,
        estimator_params=estimator_params_model,
        ood_validator_params=OODValidatorParams(),
        conformal_validator_params=ConformalValidatorParams(),
    )

    handler = CalibrateDecisionValidationCommandHandler(
        processed_data_repository=FileSystemDatasetRepository(),
        calibration_repository=FileSystemDecisionValidationCalibrationRepository(),
        estimator_factory=EstimatorFactory(),
        ood_validator_factory=OODValidatorFactory(),
        conformal_validator_factory=ConformalValidatorFactory(),
        logger=CMDLogger(name="AssuranceCalibrationLogger"),
    )

    handler.execute(command)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
