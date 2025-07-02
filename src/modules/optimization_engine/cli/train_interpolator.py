from ..application.interpolation.dtos import (
    GaussianProcessInverseDecisionMapperParams,
    KrigingInverseDecisionMapperParams,
    NearestNeighborInverseDecisoinMapperParams,
    NeuralNetworkInverserDecisionMapperParams,
    RBFInverseDecisionMapperParams,
    SplineInverseDecisionMapperParams,
    SVRInverseDecisionMapperParams,
)
from ..application.interpolation.train_single_interpolator_command.train_single_interpolator_command import (
    MetricConfig,
    NormalizerConfig,
    TrainSingleInterpolatorCommand,
)
from ..application.interpolation.train_single_interpolator_command.train_single_interpolator_handler import (
    TrainSingleInterpolatorCommandHandler,
)
from ..infrastructure.inverse_decision_mappers.factory import (
    InverseDecisionMapperFactory,
)
from ..infrastructure.loggers.cmd_logger import CMDLogger
from ..infrastructure.metrics import MetricFactory
from ..infrastructure.normalizers import NormalizerFactory
from ..infrastructure.repositories.generation.npz_pareto_data_repo import (
    NPZParetoDataRepository,
)
from ..infrastructure.repositories.interpolation.pickle_interpolator_repo import (
    PickleInterpolationModelRepository,
)
from ..infrastructure.visualizers.training_performace import (
    PlotlyTrainingPerformanceVisualizer,
)

if __name__ == "__main__":
    handler = TrainSingleInterpolatorCommandHandler(
        pareto_data_repo=NPZParetoDataRepository(),
        inverse_decision_factory=InverseDecisionMapperFactory(),
        logger=CMDLogger(name="InterpolationCMDLogger"),
        trained_model_repository=PickleInterpolationModelRepository(),
        normalizer_factory=NormalizerFactory(),
        metric_factory=MetricFactory(),
        visualizer=PlotlyTrainingPerformanceVisualizer(),
    )

    command = TrainSingleInterpolatorCommand(
        params=RBFInverseDecisionMapperParams(),
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
