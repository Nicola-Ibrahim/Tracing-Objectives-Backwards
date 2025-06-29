import time

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

    # Define the interpolator parameter classes we want to test
    # These are the DTOs for each interpolator type
    interpolator_param_classes = [
        GaussianProcessInverseDecisionMapperParams,
        NearestNeighborInverseDecisoinMapperParams,
        NeuralNetworkInverserDecisionMapperParams,
        RBFInverseDecisionMapperParams,
        KrigingInverseDecisionMapperParams,
        SVRInverseDecisionMapperParams,
        SplineInverseDecisionMapperParams,
    ]

    # Define how many times to train each interpolator type
    num_runs_per_type = 15  # Train each interpolator type 3 times

    # Common parameters for all training runs
    test_size = 0.2
    random_state = 42

    # Base name for the trained models
    base_model_name = "f1f2_vs_x1x2"

    # Loop through each interpolator type
    for param_class in interpolator_param_classes:
        interpolator_type_name = param_class.__name__.replace(
            "InverseDecisionMapperParams", ""
        )

        # Loop multiple times for each interpolator type
        for i in range(num_runs_per_type):
            run_number = i + 1
            print(
                f"  > Run {run_number}/{num_runs_per_type} for {interpolator_type_name}"
            )

            # Instantiate the parameters for the current interpolator type
            # You might want to pass specific args/kwargs if these DTOs have required params
            # For simplicity, we assume they can be instantiated without args here.
            interpolator_params_instance = param_class()

            # Create a unique base name for each run to avoid overwriting files
            # This is crucial for gathering distinct metadata files.
            unique_base_name = (
                f"{base_model_name}_{interpolator_type_name.lower()}_run{run_number}"
            )

            # Construct the command with the appropriate parameters
            command = TrainInterpolatorCommand(
                params=interpolator_params_instance,
                test_size=test_size,
                random_state=random_state,
                base_name=unique_base_name,
            )

            # Execute the command handler
            command_handler.execute(command)
            print(
                f"  Successfully completed run {run_number} for {interpolator_type_name}"
            )
            time.sleep(0.5)
