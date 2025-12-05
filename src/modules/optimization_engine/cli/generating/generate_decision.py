from ...application.generating.decision.command import (
    GenerateDecisionCommand,
)
from ...application.generating.decision.handler import (
    GenerateDecisionCommandHandler,
)
from ...domain.modeling.enums.estimator_type import EstimatorTypeEnum
from ...infrastructure.datasets.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...infrastructure.loggers.cmd_logger import CMDLogger
from ...infrastructure.modeling.repositories.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)


def main():
    """
    Main function to generate a decision using hardcoded parameters.
    Modify the variables below to change the inputs.
    """

    # Create the command object using the hardcoded values
    command = GenerateDecisionCommand(
        inverse_estimator_type=EstimatorTypeEnum.CVAE,
        forward_estimator_type=EstimatorTypeEnum.COCO,
        target_objective=[411, 1500],
        distance_tolerance=0.02,
        n_samples=50,
    )

    # Initialize the handler with pre-built services
    handler = GenerateDecisionCommandHandler(
        model_repository=FileSystemModelArtifactRepository(),
        data_repository=FileSystemDatasetRepository(),
        logger=CMDLogger(name="InterpolationCMDLogger"),
    )

    handler.execute(command)


if __name__ == "__main__":
    main()
