from ...application.factories.mertics import MetricFactory
from ...application.factories.ml_mapper import (
    MlMapperFactory,
)
from ...application.factories.normalizer import NormalizerFactory
from ...application.model_management.dtos import (
    GaussianProcessMlMapperParams,
    MDNMlMapperParams,
    NeuralNetworkMlMapperParams,
    RBFMlMapperParams,
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
    PlotlyTrainingPerformanceVisualizer,
)

if __name__ == "__main__":
    handler = TrainModelCommandHandler(
        data_repository=FileSystemDataModelRepository(),
        inverse_decision_factory=MlMapperFactory(),
        logger=CMDLogger(name="InterpolationCMDLogger"),
        trained_model_repository=FileSystemModelArtifcatRepository(),
        normalizer_factory=NormalizerFactory(),
        metric_factory=MetricFactory(),
        visualizer=None,
    )

    command = TrainModelCommand(
        ml_mapper_params=RBFMlMapperParams(),
        objectives_normalizer_config=NormalizerConfig(
            type="MinMaxScaler", params={"feature_range": (0, 1)}
        ),
        decisions_normalizer_config=NormalizerConfig(
            type="MinMaxScaler", params={"feature_range": (0, 1)}
        ),
        model_performance_metric_configs=[
            ValidationMetricConfig(type="MSE", params={}),
            ValidationMetricConfig(type="MAE", params={}),
            ValidationMetricConfig(type="R2", params={}),
        ],
        test_size=0.2,
        random_state=42,
        cv_splits=5,
    )

    # Execute the command handler
    handler.execute(command)
