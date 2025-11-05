from typing import Iterable, Mapping, Type

from ..application.dtos import (
    COCOEstimatorParams,
    CVAEEstimatorParams,
    CVAEMDNEstimatorParams,
    EstimatorParams,
    GaussianProcessEstimatorParams,
    MDNEstimatorParams,
    NeuralNetworkEstimatorParams,
    RBFEstimatorParams,
)
from ..application.factories.estimator import EstimatorFactory
from ..application.factories.mertics import MetricFactory
from ..application.modeling.train_model.train_model_command import (
    ValidationMetricConfig,
)
from ..application.modeling.train_model.train_model_handler import (
    TrainModelCommandHandler,
)
from ..infrastructure.datasets.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ..infrastructure.loggers.cmd_logger import CMDLogger
from ..infrastructure.modeling.repositories.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)

DEFAULT_VALIDATION_METRICS: tuple[str, ...] = ("MSE", "MAE", "R2")


def build_processed_training_handler() -> TrainModelCommandHandler:
    """Return a handler configured for processed dataset training runs."""

    return TrainModelCommandHandler(
        processed_data_repository=FileSystemDatasetRepository(),
        model_repository=FileSystemModelArtifactRepository(),
        logger=CMDLogger(name="InterpolationCMDLogger"),
        estimator_factory=EstimatorFactory(),
        metric_factory=MetricFactory(),
    )


def make_validation_metric_configs(
    metric_names: Iterable[str],
) -> list[ValidationMetricConfig]:
    """Create metric configs from a sequence of metric identifiers."""

    return [ValidationMetricConfig(type=name, params={}) for name in metric_names]


def create_estimator_params(
    estimator_key: str,
    overrides: Mapping[str, object] | None = None,
    *,
    registry: Mapping[str, Type[EstimatorParams]] | None = None,
) -> EstimatorParams:
    """Instantiate estimator params from the registry applying overrides."""

    registry = registry

    try:
        params_cls = registry[estimator_key]
    except KeyError as exc:  # pragma: no cover - CLI guards choices
        raise ValueError(f"Unsupported estimator '{estimator_key}'") from exc

    overrides = dict(overrides or {})
    return params_cls(**overrides) if overrides else params_cls()
