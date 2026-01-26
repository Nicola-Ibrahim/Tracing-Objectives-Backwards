import click

from modules.evaluation.application.use_cases.calibrate_decision_validation import (
    CalibrateDecisionValidationCommand,
    ConformalValidatorParams,
    OODValidatorParams,
)
from modules.evaluation.application.use_cases.calibrate_decision_validation import (
    CalibrateDecisionValidationCommandHandler,
)
from modules.modeling.application.registry import ESTIMATOR_PARAM_REGISTRY
from modules.modeling.domain.enums.estimator_type import EstimatorTypeEnum
from modules.modeling.domain.value_objects.estimator_params import EstimatorParams
from modules.evaluation.application.factories.ood_validator import (
    ConformalValidatorFactory,
    OODValidatorFactory,
)
from modules.modeling.application.factories.estimator import EstimatorFactory
from modules.evaluation.infrastructure.repositories.calibration_repository import (
    FileSystemDecisionValidationCalibrationRepository,
)
from modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from modules.shared.infrastructure.loggers.cmd_logger import CMDLogger

INVERSE_ESTIMATOR_KEYS: tuple[EstimatorTypeEnum, ...] = tuple(
    ESTIMATOR_PARAM_REGISTRY.keys()
)


def _create_estimator_params(
    estimator_key: str,
) -> EstimatorParams:
    try:
        params_cls = ESTIMATOR_PARAM_REGISTRY[EstimatorTypeEnum(estimator_key)]
    except KeyError as exc:
        raise ValueError(f"Unsupported estimator '{estimator_key}'") from exc
    return params_cls()


@click.command(help="Calibrate decision-validation gates for a trained estimator")
@click.option(
    "--estimator",
    type=click.Choice(sorted([k.value for k in INVERSE_ESTIMATOR_KEYS])),
    default=EstimatorTypeEnum.COCO.value,
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
