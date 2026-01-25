import click

from ...modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...modules.modeling.application.use_cases.generate_candidates import (
    GenerateCandidatesCommand,
    GenerateCandidatesCommandHandler,
    InverseEstimatorCandidate,
)
from ...modules.modeling.domain.enums.estimator_type import EstimatorTypeEnum
from ...modules.modeling.infrastructure.repositories.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)
from ...modules.shared.infrastructure.loggers.cmd_logger import CMDLogger


@click.command(help="Generate decision candidates for a target objective.")
def main():
    """
    Main function to generate a decision using parameters.
    """

    # Create the command object using the provided estimator and hardcoded target
    command = GenerateCandidatesCommand(
        dataset_name="cocoex_f5",
        inverse_estimator=InverseEstimatorCandidate(type=EstimatorTypeEnum.MDN, version=1),
        forward_estimator_type=EstimatorTypeEnum.COCO,
        target_objective=[410, 1400],
        distance_tolerance=0.02,
        n_samples=20,
        diversity_method="euclidean",
    )

    # Initialize the handler with pre-built services
    handler = GenerateCandidatesCommandHandler(
        model_repository=FileSystemModelArtifactRepository(),
        data_repository=FileSystemDatasetRepository(),
        logger=CMDLogger(name="InterpolationCMDLogger"),
    )

    handler.execute(command)


if __name__ == "__main__":
    main()
