from ...application.modeling.generate_decision.generate_decision_command import (
    GenerateDecisionCommand,
)
from ...application.modeling.generate_decision.generate_decision_handler import (
    GenerateDecisionCommandHandler,
)
from ...domain.modeling.enums.estimator_type_enum import (
    EstimatorTypeEnum,
)
from ...infrastructure.loggers.cmd_logger import CMDLogger
from ...infrastructure.repositories.datasets.generated_dataset_repo import (
    FileSystemGeneratedDatasetRepository,
)
from ...infrastructure.repositories.modeling.model_artifact_repo import (
    FileSystemModelArtifcatRepository,
)


def main():
    """
    Main function to generate a decision using hardcoded parameters.
    Modify the variables below to change the inputs.
    """

    # Specify the target objective point (f1, f2, ...).
    target_objective_point = [411, 1242]

    # Initialize the handler with the repository
    handler = GenerateDecisionCommandHandler(
        interpolation_model_repo=FileSystemModelArtifcatRepository(),
        data_repository=FileSystemGeneratedDatasetRepository(),
        logger=CMDLogger(name="InterpolationCMDLogger"),
    )

    # Create the command object using the hardcoded values
    command = GenerateDecisionCommand(
        model_type=EstimatorTypeEnum.KRIGING_ND.value,
        target_objective=target_objective_point,
        distance_tolerance=0.02,
        num_suggestions=5,
    )

    handler.execute(command)


if __name__ == "__main__":
    main()
