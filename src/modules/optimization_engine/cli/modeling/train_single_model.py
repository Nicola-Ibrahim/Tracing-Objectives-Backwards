from ...application.factories.estimator import (
    EstimatorFactory,
)
from ...application.factories.mertics import MetricFactory
from ...application.factories.normalizer import NormalizerFactory
from ...application.model_management.dtos import (
    CVAEEstimatorParams,
    GaussianProcessEstimatorParams,
    MDNEstimatorParams,
    NeuralNetworkEstimatorParams,
    RBFEstimatorParams,
)
from ...application.model_management.train_model.train_model_command import (
    NormalizerConfig,
    TrainModelCommand,
    ValidationMetricConfig,
)
from ...application.model_management.train_model.train_model_handler import (
    TrainModelCommandHandler,
)
from ...infrastructure.loggers.cmd_logger import CMDLogger
from ...infrastructure.repositories.generation.data_model_repo import (
    FileSystemDataModelRepository,
)
from ...infrastructure.repositories.model_management.model_artifact_repo import (
    FileSystemModelArtifcatRepository,
)
from ...infrastructure.visualizers.training_performace import (
    LearningCurveVisualizer,
)

if __name__ == "__main__":
    handler = TrainModelCommandHandler(
        data_repository=FileSystemDataModelRepository(),
        model_repository=FileSystemModelArtifcatRepository(),
        logger=CMDLogger(name="InterpolationCMDLogger"),
        normalizer_factory=NormalizerFactory(),
        estimator_factory=EstimatorFactory(),
        metric_factory=MetricFactory(),
        visualizer=LearningCurveVisualizer(
            metric=MetricFactory().create(config={"type": "nll", "params": {}})
        ),
    )

    command = TrainModelCommand(
        estimator_params=MDNEstimatorParams(),
        normalizer_config=NormalizerConfig(
            type="MinMaxScaler", params={"feature_range": (0, 1)}
        ),
        model_performance_metric_configs=[
            ValidationMetricConfig(type="MSE", params={}),
            ValidationMetricConfig(type="MAE", params={}),
            ValidationMetricConfig(type="R2", params={}),
        ],
        test_size=0.2,
        random_state=42,
        cv_splits=10,
    )

    # Execute the command handler
    handler.execute(command)
