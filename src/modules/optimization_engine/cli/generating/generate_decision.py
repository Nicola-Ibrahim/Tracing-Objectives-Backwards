import click

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


@click.command(help="Generate decision candidates for a target objective.")
@click.option(
    "--estimator",
    type=click.Choice([e.value for e in EstimatorTypeEnum]),
    default=EstimatorTypeEnum.MDN.value,
    show_default=True,
    help="Inverse estimator type to use.",
)
def main(estimator: str):
    """
    Main function to generate a decision using parameters.
    """

    # Create the command object using the provided estimator and hardcoded target
    command = GenerateDecisionCommand(
        inverse_estimator_type=EstimatorTypeEnum(estimator),
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
