import click

from ...application.assuring.calibrate_decision_validation.command import (
    CalibrateDecisionValidationCommand,
    ConformalValidatorParams,
    OODValidatorParams,
)
from ...application.assuring.calibrate_decision_validation.handler import (
    CalibrateDecisionValidationCommandHandler,
)
from ...application.training.registry import ESTIMATOR_PARAM_REGISTRY
from ...domain.modeling.enums.estimator_key import EstimatorKeyEnum
from ...domain.modeling.value_objects.estimator_params import EstimatorParams
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

INVERSE_ESTIMATOR_KEYS: tuple[EstimatorKeyEnum, ...] = tuple(
    ESTIMATOR_PARAM_REGISTRY.keys()
)


def _create_estimator_params(
    estimator_key: str,
) -> EstimatorParams:
    try:
        params_cls = ESTIMATOR_PARAM_REGISTRY[EstimatorKeyEnum(estimator_key)]
    except KeyError as exc:
        raise ValueError(f"Unsupported estimator '{estimator_key}'") from exc
    return params_cls()


@click.command(help="Calibrate decision-validation gates for a trained estimator")
@click.option(
    "--estimator",
    type=click.Choice(sorted([k.value for k in INVERSE_ESTIMATOR_KEYS])),
    default=EstimatorKeyEnum.COCO.value,
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
        ood_validator_params=OODValidatorParams(
            method="mahalanobis",
            percentile=97.5,
            cov_reg=1e-6,
        ),
        conformal_validator_params=ConformalValidatorParams(
            method="split_conformal_l2",
            confidence=0.90,
        ),
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
