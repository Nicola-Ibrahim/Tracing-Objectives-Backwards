import ast
from typing import Mapping, Sequence, Type

import click

from ...application.dtos import (
    COCOEstimatorParams,
    CVAEEstimatorParams,
    CVAEMDNEstimatorParams,
    EstimatorParams,
    GaussianProcessEstimatorParams,
    MDNEstimatorParams,
    NeuralNetworkEstimatorParams,
    RBFEstimatorParams,
    ValidationMetricConfig,
)
from ...application.factories.estimator import EstimatorFactory
from ...application.factories.mertics import MetricFactory
from ...application.modeling.train_inverse_model import (
    TrainInverseModelCommand,
    TrainInverseModelCommandHandler,
)
from ...application.modeling.train_inverse_model_cv import (
    TrainInverseModelCrossValidationCommand,
    TrainInverseModelCrossValidationCommandHandler,
)
from ...application.modeling.train_inverse_model_grid_search import (
    TrainInverseModelGridSearchCommand,
    TrainInverseModelGridSearchCommandHandler,
)
from ...infrastructure.datasets.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...infrastructure.loggers.cmd_logger import CMDLogger
from ...infrastructure.modeling.repositories.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)

DEFAULT_VALIDATION_METRICS: tuple[str, ...] = ("MSE", "MAE", "R2")
INVERSE_ESTIMATOR_REGISTRY: dict[str, Type[EstimatorParams]] = {
    "cvae": CVAEEstimatorParams,
    "cvae_mdn": CVAEMDNEstimatorParams,
    "gaussian_process": GaussianProcessEstimatorParams,
    "mdn": MDNEstimatorParams,
    "neural_network": NeuralNetworkEstimatorParams,
    "rbf": RBFEstimatorParams,
    "coco": COCOEstimatorParams,
}


def _literal(value: str):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _parse_overrides(pairs: Sequence[str]) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise click.BadParameter(
                f"Parameter override '{pair}' must be in the form key=value",
                param_hint="--param",
            )
        key, raw_value = pair.split("=", 1)
        overrides[key.strip()] = _literal(raw_value.strip())
    return overrides


def _create_estimator_params(
    estimator_key: str,
    overrides: Mapping[str, object] | None = None,
) -> EstimatorParams:
    try:
        params_cls = INVERSE_ESTIMATOR_REGISTRY[estimator_key]
    except KeyError as exc:  # pragma: no cover - CLI guards choices
        raise ValueError(f"Unsupported estimator '{estimator_key}'") from exc

    overrides = dict(overrides or {})
    return params_cls(**overrides) if overrides else params_cls()


def _make_validation_metric_configs(
    metric_names: Sequence[str],
) -> list[ValidationMetricConfig]:
    return [ValidationMetricConfig(type=name, params={}) for name in metric_names]


def _build_inverse_training_handler() -> TrainInverseModelCommandHandler:
    return TrainInverseModelCommandHandler(
        processed_data_repository=FileSystemDatasetRepository(),
        model_repository=FileSystemModelArtifactRepository(),
        logger=CMDLogger(name="InterpolationCMDLogger"),
        estimator_factory=EstimatorFactory(),
        metric_factory=MetricFactory(),
    )


def _build_inverse_cv_handler() -> TrainInverseModelCrossValidationCommandHandler:
    return TrainInverseModelCrossValidationCommandHandler(
        processed_data_repository=FileSystemDatasetRepository(),
        model_repository=FileSystemModelArtifactRepository(),
        logger=CMDLogger(name="InterpolationCVCMDLogger"),
        estimator_factory=EstimatorFactory(),
        metric_factory=MetricFactory(),
    )


def _build_inverse_grid_handler() -> TrainInverseModelGridSearchCommandHandler:
    return TrainInverseModelGridSearchCommandHandler(
        processed_data_repository=FileSystemDatasetRepository(),
        model_repository=FileSystemModelArtifactRepository(),
        logger=CMDLogger(name="InterpolationGridCMDLogger"),
        estimator_factory=EstimatorFactory(),
        metric_factory=MetricFactory(),
    )


def _common_training_options(func):
    options = [
        click.option(
            "--estimator",
            type=click.Choice(sorted(INVERSE_ESTIMATOR_REGISTRY.keys())),
            default=next(iter(INVERSE_ESTIMATOR_REGISTRY)),
            show_default=True,
            help="Which estimator configuration to use",
        ),
        click.option(
            "--param",
            "params",
            multiple=True,
            metavar="KEY=VALUE",
            help="Override estimator parameter (may be supplied multiple times)",
        ),
        click.option(
            "--metric",
            "metrics",
            multiple=True,
            default=DEFAULT_VALIDATION_METRICS,
            show_default=True,
            help="Validation metrics to report",
        ),
        click.option(
            "--random-state",
            type=int,
            default=42,
            show_default=True,
            help="Random seed for reproducibility",
        ),
        click.option(
            "--learning-curve-steps",
            type=int,
            default=50,
            show_default=True,
            help="Number of steps for deterministic learning curves",
        ),
        click.option(
            "--epochs",
            type=int,
            default=100,
            show_default=True,
            help="Epochs for probabilistic estimators",
        ),
    ]

    for option in reversed(options):
        func = option(func)
    return func


__all__ = [
    "_build_inverse_training_handler",
    "_build_inverse_cv_handler",
    "_build_inverse_grid_handler",
    "_common_training_options",
    "_create_estimator_params",
    "_literal",
    "_make_validation_metric_configs",
    "_parse_overrides",
    "DEFAULT_VALIDATION_METRICS",
    "INVERSE_ESTIMATOR_REGISTRY",
    "TrainInverseModelCommand",
    "TrainInverseModelCommandHandler",
    "TrainInverseModelCrossValidationCommand",
    "TrainInverseModelCrossValidationCommandHandler",
    "TrainInverseModelGridSearchCommand",
    "TrainInverseModelGridSearchCommandHandler",
]
