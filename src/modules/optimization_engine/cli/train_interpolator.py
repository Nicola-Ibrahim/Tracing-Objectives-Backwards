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
from ..infrastructure.metrics import MeanSquaredErrorValidationMetric
from ..infrastructure.normalizers import MinMaxScalerNormalizer
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
        decision_mapper_training_service=DecisionMapperTrainingService(
            validation_metric=MeanSquaredErrorValidationMetric(),
            objectives_normalizer=MinMaxScalerNormalizer(),
            decisions_normalizer=MinMaxScalerNormalizer(),
        ),
        trained_model_repository=PickleInterpolationModelRepository(),
    )

    # Common parameters for all training runs
    test_size = 0.2
    random_state = 42

    # Construct the command with the appropriate parameters
    command = TrainInterpolatorCommand(
        params=RBFInverseDecisionMapperParams(),
        test_size=test_size,
        random_state=random_state,
        version_number=16,
        should_generate_plots=True,
    )

    # Execute the command handler
    command_handler.execute(command)
