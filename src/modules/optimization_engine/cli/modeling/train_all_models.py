import time

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
    FileSystemModelArtifcatRepository,
)

if __name__ == "__main__":
    # Initialize the command handler once, as its dependencies are fixed
    command_handler = TrainModelCommandHandler(
        data_repository=NPZParetoDataRepository(),
        inverse_decision_factory=InverseDecisionMapperFactory(),
        logger=CMDLogger(name="InterpolationCMDLogger"),
        trained_model_repository=FileSystemModelArtifcatRepository(),
        normalizer_factory=NormalizerFactory(),
        metric_factory=MetricFactory(),
    )

    # Define the model parameter classes we want to test
    # These are the DTOs for each model type
    model_param_classes = [
        GaussianProcessInverseDecisionMapperParams,
        NeuralNetworkInverseDecisionMapperParams,
        RBFInverseDecisionMapperParams,
        MDNInverseDecisionMapperParams,
    ]

    # Define how many times to train each interpolator type
    num_runs_per_type = 15  # Train each interpolator type 3 times

    # Loop through each model type
    for param_class in model_param_classes:
        model_type_name = param_class.__name__.replace(
            "InverseDecisionMapperParams", ""
        )

        # Loop multiple times for each model type
        for i in range(num_runs_per_type):
            version_number = i + 1
            print(
                f"  > Run {version_number}/{num_runs_per_type} for { model_type_name}"
            )

            # Instantiate the parameters for the current interpolator type
            # You might want to pass specific args/kwargs if these DTOs have required params
            # For simplicity, we assume they can be instantiated without args here.
            model_params_instance = param_class()

            # Construct the command with the appropriate parameters
            command = TrainModelCommand(
                inverse_decision_mapper_params=model_params_instance,
                # --- NEW NORMALIZER & METRIC CONFIGURATIONS ---
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
            )

            # Execute the command handler
            command_handler.execute(command)
            print(
                f"  Successfully completed run {version_number} for { model_type_name}"
            )
            time.sleep(0.8)
