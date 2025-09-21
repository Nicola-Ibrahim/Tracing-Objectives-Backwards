from ...application.modeling.generate_decision.generate_decision_command import (
    GenerateDecisionCommand,
)
from ...application.modeling.generate_decision.generate_decision_handler import (
    GenerateDecisionCommandHandler,
)
from ...domain.modeling.enums.estimator_type import (
    EstimatorTypeEnum,
)
from ...infrastructure.loggers.cmd_logger import CMDLogger
from ...infrastructure.repositories.datasets.processed_dataset_repo import (
    FileSystemProcessedDatasetRepository,
)
from ...infrastructure.repositories.modeling.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)
from ...application.factories.estimator import EstimatorFactory
from ...application.factories.assurance import (
    create_default_scoring_strategy,
    create_default_diversity_registry,
)


def main():
    """
    Main function to generate a decision using hardcoded parameters.
    Modify the variables below to change the inputs.
    """

    # Specify the target objective point (f1, f2, ...).
    target_objective_point = [411, 1242]

    # Build forward model via EstimatorFactory (configurable)
    forward_model = EstimatorFactory().create_forward(
        {"type": "coco_biobj", "function_indices": 5}
    )

    # Initialize the handler with the repositories and forward model
    handler = GenerateDecisionCommandHandler(
        model_repository=FileSystemModelArtifactRepository(),
        processed_data_repository=FileSystemProcessedDatasetRepository(),
        logger=CMDLogger(name="InterpolationCMDLogger"),
        forward_model=forward_model,
        scoring_factory=create_default_scoring_strategy,
        diversity_registry_factory=create_default_diversity_registry,
    )

    # Create the command object using the hardcoded values
    command = GenerateDecisionCommand(
        estimator_type=EstimatorTypeEnum.KRIGING_ND.value,
        target_objective=target_objective_point,
        distance_tolerance=0.02,
        num_suggestions=5,
    )

    handler.execute(command)


if __name__ == "__main__":
    main()
