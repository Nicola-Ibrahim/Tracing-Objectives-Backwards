import time

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

if __name__ == "__main__":
    # Initialize the command handler once, as its dependencies are fixed
    command_handler = TrainSingleModelCommandHandler(
        pareto_data_repo=NPZParetoDataRepository(),
        inverse_decision_factory=InverseDecisionMapperFactory(),
        logger=CMDLogger(name="InterpolationCMDLogger"),
        trained_model_repository=PickleInterpolationModelRepository(),
        normalizer_factory=NormalizerFactory(),
        metric_factory=MetricFactory(),
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
        MDNInverseDecisionMapperParams,
    ]

    # Define how many times to train each interpolator type
    num_runs_per_type = 15  # Train each interpolator type 3 times

    # Loop through each interpolator type
    for param_class in interpolator_param_classes:
        interpolator_type_name = param_class.__name__.replace(
            "InverseDecisionMapperParams", ""
        )

        # Loop multiple times for each interpolator type
        for i in range(num_runs_per_type):
            version_number = i + 1
            print(
                f"  > Run {version_number}/{num_runs_per_type} for {interpolator_type_name}"
            )

            # Instantiate the parameters for the current interpolator type
            # You might want to pass specific args/kwargs if these DTOs have required params
            # For simplicity, we assume they can be instantiated without args here.
            interpolator_params_instance = param_class()

            # Construct the command with the appropriate parameters
            command = TrainSingleModelCommand(
                params=interpolator_params_instance,
                version_number=version_number,
                # --- NEW NORMALIZER & METRIC CONFIGURATIONS ---
                objectives_normalizer_config=NormalizerConfig(
                    type="MinMaxScaler",
                    params={"feature_range": (0, 1)},
                ),
                decisions_normalizer_config=NormalizerConfig(
                    type="MinMaxScaler",
                    params={"feature_range": (0, 1)},
                ),
                validation_metric_config=MetricConfig(
                    type="MSE",
                    params={},  # No specific params for MSE
                ),
            )

            # Execute the command handler
            command_handler.execute(command)
            print(
                f"  Successfully completed run {version_number} for {interpolator_type_name}"
            )
            time.sleep(1)
