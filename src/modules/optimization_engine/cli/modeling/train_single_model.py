from ...application.dtos import (
    CVAEEstimatorParams,
    CVAEMDNEstimatorParams,
    GaussianProcessEstimatorParams,
    MDNEstimatorParams,
    NeuralNetworkEstimatorParams,
    RBFEstimatorParams,
)
from ...application.factories.estimator import (
    EstimatorFactory,
)
from ...application.factories.mertics import MetricFactory
from ...application.modeling.train_model.train_model_command import (
    TrainModelCommand,
    ValidationMetricConfig,
)
from ...application.modeling.train_model.train_model_handler import (
    TrainModelCommandHandler,
)
from ...infrastructure.loggers.cmd_logger import CMDLogger
from ...infrastructure.repositories.datasets.processed_dataset_repo import (
    FileSystemProcessedDatasetRepository,
)
from ...infrastructure.repositories.modeling.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)

if __name__ == "__main__":
    handler = TrainModelCommandHandler(
        processed_data_repository=FileSystemProcessedDatasetRepository(),
        model_repository=FileSystemModelArtifactRepository(),
        logger=CMDLogger(name="InterpolationCMDLogger"),
        estimator_factory=EstimatorFactory(),
        metric_factory=MetricFactory(),
    )

    command = TrainModelCommand(
        estimator_params=CVAEEstimatorParams(),
        model_performance_metric_configs=[
            ValidationMetricConfig(type="MSE", params={}),
            ValidationMetricConfig(type="MAE", params={}),
            ValidationMetricConfig(type="R2", params={}),
        ],
        test_size=0.2,
        random_state=42,
        learning_curve_steps=50,
        # cv_splits=10,
        # tune_param_name="n_neighbors",
        # tune_param_range=[5, 10, 20, 50],
    )

    handler.execute(command)
