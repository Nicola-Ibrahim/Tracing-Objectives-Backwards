import ast
from typing import Sequence

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
from ...infrastructure.datasets.repositories.processed_dataset_repo import (
    FileSystemProcessedDatasetRepository,
)
from ...infrastructure.loggers.cmd_logger import CMDLogger
from ..common import ESTIMATOR_REGISTRY, create_estimator_params


def _literal(value: str):
    """Safely coerce CLI string values into Python literals when possible."""

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _parse_overrides(pairs: Sequence[str]) -> dict[str, object]:
    """Convert ``key=value`` pairs passed on the CLI into kwargs."""

    overrides: dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise click.BadParameter(
                f"Parameter override '{pair}' must be in the form key=value",
                param_hint="--estimator-param",
            )
        key, raw_value = pair.split("=", 1)
        overrides[key.strip()] = _literal(raw_value.strip())
    return overrides


def build_calibration_handler() -> CalibrateDecisionValidationCommandHandler:
    """Return a command handler configured for filesystem-backed calibration."""

    return CalibrateDecisionValidationCommandHandler(
        processed_data_repository=FileSystemProcessedDatasetRepository(),
        calibration_repository=FileSystemDecisionValidationCalibrationRepository(),
        estimator_factory=EstimatorFactory(),
        ood_calibrator_factory=OODCalibratorFactory(),
        conformal_calibrator_factory=ConformalCalibratorFactory(),
        logger=CMDLogger(name="AssuranceCalibrationLogger"),
    )


@click.command(help="Calibrate decision-validation gates for a trained estimator")
@click.option(
    "--dataset",
    "dataset_name",
    type=str,
    default="dataset",
    show_default=True,
    help="Name of the processed dataset to use",
)
@click.option(
    "--estimator",
    type=click.Choice(sorted(ESTIMATOR_REGISTRY.keys())),
    default="coco",
    show_default=True,
    help="Estimator configuration used for calibration",
)
@click.option(
    "--estimator-param",
    "estimator_params",
    multiple=True,
    metavar="KEY=VALUE",
    help="Override estimator parameter (may be supplied multiple times)",
)
@click.option(
    "--ood-percentile",
    type=float,
    default=97.5,
    show_default=True,
    help="Percentile for the OOD threshold",
)
@click.option(
    "--ood-cov-reg",
    type=float,
    default=1e-6,
    show_default=True,
    help="Covariance regularisation for Mahalanobis OOD calibrator",
)
@click.option(
    "--conformal-confidence",
    type=float,
    default=0.90,
    show_default=True,
    help="Confidence level for split-conformal calibration",
)
def cli(
    dataset_name: str,
    estimator: str,
    estimator_params: Sequence[str],
    ood_percentile: float,
    ood_cov_reg: float,
    conformal_confidence: float,
) -> None:
    handler = build_calibration_handler()
    overrides = _parse_overrides(estimator_params)
    estimator_params_model = create_estimator_params(estimator, overrides)

    command = CalibrateDecisionValidationCommand(
        dataset_name=dataset_name,
        estimator_params=estimator_params_model,
        ood_calibrator_params=OODCalibratorParams(
            percentile=ood_percentile,
            cov_reg=ood_cov_reg,
        ),
        conformal_calibrator_params=ConformalCalibratorParams(
            confidence=conformal_confidence,
        ),
    )
    handler.execute(command)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
