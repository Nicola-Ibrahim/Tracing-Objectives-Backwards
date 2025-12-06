import click

from ...application.generating.generate_decision.command import (
    GenerateDecisionCommand,
)
from ...application.generating.generate_decision.handler import (
    GenerateDecisionCommandHandler,
)
from ...domain.modeling.enums.estimator_type import EstimatorTypeEnum
from ...domain.modeling.services.decision_generation import DecisionGenerator
from ...infrastructure.datasets.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...infrastructure.loggers.cmd_logger import CMDLogger
from ...infrastructure.modeling.repositories.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)


@click.command(help="Generate decision candidates for a target objective.")
def main():
    """
    Main function to generate a decision using parameters.
    """

    # Create the command object using the provided estimator and hardcoded target
    command = GenerateDecisionCommand(
        inverse_estimator_types=[EstimatorTypeEnum.MDN, EstimatorTypeEnum.CVAE],
        forward_estimator_type=EstimatorTypeEnum.COCO,
        target_objective=[408, 1282],
        distance_tolerance=0.02,
        n_samples=50,
    )

    # Initialize the handler with pre-built services
    handler = GenerateDecisionCommandHandler(
        model_repository=FileSystemModelArtifactRepository(),
        data_repository=FileSystemDatasetRepository(),
        logger=CMDLogger(name="InterpolationCMDLogger"),
        generator_service=DecisionGenerator(),
    )

    handler.execute(command)


if __name__ == "__main__":
    main()
