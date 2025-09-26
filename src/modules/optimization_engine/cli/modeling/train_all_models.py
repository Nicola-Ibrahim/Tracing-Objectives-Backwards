import time

from ...application.dtos import (
    EstimatorParams,
    GaussianProcessEstimatorParams,
    MDNEstimatorParams,
    NeuralNetworkEstimatorParams,
    RBFEstimatorParams,
)
from ...application.factories.estimator import EstimatorFactory
from ...application.factories.mertics import MetricFactory
from ...application.factories.normalizer import NormalizerFactory
from ...application.modeling.train_model.train_model_command import TrainModelCommand
from ...application.modeling.train_model.train_model_handler import (
    TrainModelCommandHandler,
)
from ...infrastructure.loggers.cmd_logger import CMDLogger
from ...infrastructure.repositories.datasets.generated_dataset_repo import (
    FileSystemGeneratedDatasetRepository,
)
from ...infrastructure.repositories.repositories.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)
from .common import (
    DEFAULT_VALIDATION_METRICS,
    make_validation_metric_configs,
)


def _build_command(param_cls: type[EstimatorParams]) -> TrainModelCommand:
    """Assemble a TrainModelCommand for the provided estimator parameter class."""

    params_instance = param_cls()
    metric_configs = make_validation_metric_configs(DEFAULT_VALIDATION_METRICS)

    return TrainModelCommand(
        estimator_params=params_instance,
        estimator_performance_metric_configs=metric_configs,
    )


if __name__ == "__main__":
    # Initialize the command handler once, as its dependencies are fixed
    command_handler = TrainModelCommandHandler(
        data_repository=FileSystemGeneratedDatasetRepository(),
        estimator_factory=EstimatorFactory(),
        logger=CMDLogger(name="InterpolationCMDLogger"),
        model_repository=FileSystemModelArtifactRepository(),
        normalizer_factory=NormalizerFactory(),
        metric_factory=MetricFactory(),
    )

    # Define the model parameter classes we want to test
    # These are the DTOs for each model type
    model_param_classes = [
        GaussianProcessEstimatorParams,
        NeuralNetworkEstimatorParams,
        RBFEstimatorParams,
        MDNEstimatorParams,
    ]

    # Define how many times to train each interpolator type
    num_runs_per_type = 15  # Train each interpolator type 3 times

    # Loop through each model type
    for param_class in model_param_classes:
        estimator_type_name = param_class.__name__.replace("EstimatorParams", "")

        # Loop multiple times for each model type
        for run_idx in range(num_runs_per_type):
            version_number = run_idx + 1
            print(
                f"  > Run {version_number}/{num_runs_per_type} for {estimator_type_name}"
            )

            command = _build_command(param_class)

            # Execute the command handler
            command_handler.execute(command)
            print(
                f"  Successfully completed run {version_number} for {estimator_type_name}"
            )
            time.sleep(0.8)
