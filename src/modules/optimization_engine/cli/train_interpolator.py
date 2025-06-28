from ..application.interpolation.train_model.dtos import (
    CloughTocherInverseDecisionMapperParams,
    LinearInverseDecisionMapperParams,
    NearestNeighborInverseDecisoinMapperParams,
    RBFInverseDecisionMapperParams,
)
from ..application.interpolation.train_model.train_interpolator_command import (
    TrainInterpolatorCommand,
)
from ..application.interpolation.train_model.train_interpolator_handler import (
    TrainInterpolatorCommandHandler,
)
from ..domain.services.training_service import DecisionMapperTrainingService
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


command = TrainInterpolatorCommand(
    params=RBFInverseDecisionMapperParams(),
    test_size=0.2,
    random_state=42,
    base_name="f1f2_vs_x1x2",
)

command_handler.execute(command)
