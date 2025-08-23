from ...application.factories.inverse_decision_mapper import (
    InverseDecisionMapperFactory,
)
from ...application.factories.mertics import MetricFactory
from ...application.factories.normalizer import NormalizerFactory
from ...application.model_management.dtos import (
    GaussianProcessInverseDecisionMapperParams,
    MDNInverseDecisionMapperParams,
    NeuralNetworkInverseDecisionMapperParams,
    RBFInverseDecisionMapperParams,
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
from ...infrastructure.repositories.generation.npz_pareto_data_repo import (
    NPZParetoDataRepository,
)
from ...infrastructure.repositories.model_management.pickle_model_artifact_repo import (
    PickleInterpolationModelRepository,
)
from ...infrastructure.visualizers.training_performace import (
    PlotlyTrainingPerformanceVisualizer,
)

if __name__ == "__main__":
    handler = TrainModelCommandHandler(
        data_repository=NPZParetoDataRepository(),
        inverse_decision_factory=InverseDecisionMapperFactory(),
        logger=CMDLogger(name="InterpolationCMDLogger"),
        trained_model_repository=PickleInterpolationModelRepository(),
        normalizer_factory=NormalizerFactory(),
        metric_factory=MetricFactory(),
        visualizer=None,
    )

    command = TrainModelCommand(
        inverse_decision_mapper_params=GaussianProcessInverseDecisionMapperParams(),
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
