from ..application.interpolation.train_model.dtos import (
    GaussianProcessInverseDecisionMapperParams,
    KrigingInverseDecisionMapperParams,
    NearestNeighborInverseDecisoinMapperParams,
    NeuralNetworkInverserDecisionMapperParams,
    RBFInverseDecisionMapperParams,
    SplineInverseDecisionMapperParams,
    SVRInverseDecisionMapperParams,
)
from ..application.interpolation.train_model.train_interpolator_command import (
    MetricConfig,
    NormalizerConfig,
    TrainInterpolatorCommand,
)
from ..application.interpolation.train_model.train_interpolator_handler import (
    TrainInterpolatorCommandHandler,
)
from ..domain.services.training import DecisionMapperTrainingService
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

if __name__ == "__main__":
    # Initialize the command handler once, as its dependencies are fixed
    command_handler = TrainInterpolatorCommandHandler(
        pareto_data_repo=NPZParetoDataRepository(),
        inverse_decision_factory=InverseDecisionMapperFactory(),
        logger=CMDLogger(name="InterpolationCMDLogger"),
        decision_mapper_training_service=DecisionMapperTrainingService(),
        trained_model_repository=PickleInterpolationModelRepository(),
        normalizer_factory=NormalizerFactory(),
        metric_factory=MetricFactory(),
    )

    # Common parameters for all training runs
    test_size = 0.2
    random_state = 42

    # Construct the command with the appropriate parameters
    command = TrainInterpolatorCommand(
        params=RBFInverseDecisionMapperParams(),
        test_size=test_size,
        random_state=random_state,
        version_number=1,
        should_generate_plots=True,
        # --- NEW NORMALIZER & METRIC CONFIGS ---
        objectives_normalizer_config=NormalizerConfig(
            type="MinMaxScaler", params={"feature_range": (0, 1)}
        ),
        decisions_normalizer_config=NormalizerConfig(
            type="MinMaxScaler", params={"feature_range": (0, 1)}
        ),
        validation_metric_config=MetricConfig(type="MSE"),
    )

    # Execute the command handler
    command_handler.execute(command)
