from ..application.interpolation.train_model.dtos import (
    CloughTocherInterpolatorParams,
    LinearInterpolatorParams,
    RBFInterpolatorParams,
)
from ..application.interpolation.train_model.train_interpolator_command import (
    TrainInterpolatorCommand,
)
from ..application.interpolation.train_model.train_interpolator_handler import (
    TrainInterpolatorCommandHandler,
)
from ..domain.interpolation.entities.inverse_decision_mapper_type import (
    InverseDecisionMapperType,
)
from ..domain.services.training_service import DecisionMapperTrainingService
from ..infrastructure.inverse_decision_mappers.factory import (
    InverseDecisionMapperFactory,
)
from ..infrastructure.loggers.cmd_logger import CMDLogger
from ..infrastructure.metrics import MeanSquaredErrorValidationMetric
from ..infrastructure.normalizers import MinMaxScalerNormalizer
from ..infrastructure.repositories.npz_pareto_data_repo import (
    NPZParetoDataRepository,
)
from ..infrastructure.repositories.pickle_interpolator_repo import (
    PickleInterpolationModelRepository,
)

# Instantiate the TrainInterpolatorCommandHandler
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


linear_command = TrainInterpolatorCommand(
    data_file_name="pareto_data",
    model_conceptual_name="My1DLinearMapper",
    type=InverseDecisionMapperType.RBF_ND,
    params=RBFInterpolatorParams(),
    test_size=0.2,
    random_state=42,
    description="1D Linear Interpolator trained on synthetic data for demonstration.",
    notes="This is a test run to verify the full dependency injection setup.",
    collection="Demo_1D_Mappers",
    training_data_identifier="synthetic_1d_data_v1",
)

try:
    command_handler.handle(linear_command)
    print("\nLinear Interpolator training command executed successfully.")


except Exception as e:
    print(f"\nAn error occurred during command execution: {e}")
    import traceback

    traceback.print_exc()
