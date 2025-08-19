from ..application.factories.inverse_decision_mapper import (
    InverseDecisionMapperFactory,
)
from ..application.model_management.dtos import (
    GaussianProcessInverseDecisionMapperParams,
    KrigingInverseDecisionMapperParams,
    MDNInverseDecisionMapperParams,
    NearestNeighborInverseDecisoinMapperParams,
    NeuralNetworkInverserDecisionMapperParams,
    RBFInverseDecisionMapperParams,
    SplineInverseDecisionMapperParams,
    SVRInverseDecisionMapperParams,
)
from ..application.model_management.train_model.train_single_model_command import (
    MetricConfig,
    NormalizerConfig,
    TrainSingleModelCommand,
)
from ..application.model_management.train_model.train_single_model_handler import (
    TrainSingleModelCommandHandler,
)
from ..infrastructure.loggers.cmd_logger import CMDLogger
from ..infrastructure.metrics import MetricFactory
from ..infrastructure.normalizers import NormalizerFactory
from ..infrastructure.repositories.generation.npz_pareto_data_repo import (
    NPZParetoDataRepository,
)
from ..infrastructure.repositories.model_management.pickle_model_artifact_repo import (
    PickleInterpolationModelRepository,
)
from ..infrastructure.visualizers.training_performace import (
    PlotlyTrainingPerformanceVisualizer,
)

if __name__ == "__main__":
    handler = TrainSingleModelCommandHandler(
        pareto_data_repo=NPZParetoDataRepository(),
        inverse_decision_factory=InverseDecisionMapperFactory(),
        logger=CMDLogger(name="InterpolationCMDLogger"),
        trained_model_repository=PickleInterpolationModelRepository(),
        normalizer_factory=NormalizerFactory(),
        metric_factory=MetricFactory(),
        visualizer=PlotlyTrainingPerformanceVisualizer(),
    )

    command = TrainSingleModelCommand(
        params=MDNInverseDecisionMapperParams(),
        version_number=1,
        should_generate_plots=True,
        objectives_normalizer_config=NormalizerConfig(
            type="MinMaxScaler", params={"feature_range": (0, 1)}
        ),
        decisions_normalizer_config=NormalizerConfig(
            type="MinMaxScaler", params={"feature_range": (0, 1)}
        ),
        validation_metric_config=MetricConfig(type="MSE"),
    )

    # Execute the command handler
    handler.execute(command)
