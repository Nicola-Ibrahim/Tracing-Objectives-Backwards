import click

from ...application.assuring.compare_inverse_models.command import ModelCandidate
from ...application.generating.generate_decision.command import (
    GenerateDecisionCommand,
)
from ...application.generating.generate_decision.handler import (
    GenerateDecisionCommandHandler,
)
from ...domain.modeling.enums.estimator_type import EstimatorTypeEnum
from ...infrastructure.datasets.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...infrastructure.modeling.repositories.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)
from ...infrastructure.shared.loggers.cmd_logger import CMDLogger
from ...infrastructure.visualization.decision_generation import (
    DecisionGenerationComparisonVisualizer,
)
from ...workflows.decision_generation_workflow import DecisionGenerationWorkflow


@click.command(help="Generate decision candidates for a target objective.")
@click.option(
    "--dataset-name",
    default="dataset",
    show_default=True,
    help="Dataset identifier to load for decision generation.",
)
def main(dataset_name: str):
    """
    Main function to generate a decision using parameters.
    """

    # Create the command object using the provided estimator and hardcoded target
    command = GenerateDecisionCommand(
        dataset_name=dataset_name,
        inverse_candidates=[
            ModelCandidate(type=EstimatorTypeEnum.MDN, version="1"),
            ModelCandidate(type=EstimatorTypeEnum.MDN, version="2"),
            ModelCandidate(type=EstimatorTypeEnum.MDN, version="3"),
            ModelCandidate(type=EstimatorTypeEnum.MDN, version="4"),
            ModelCandidate(type=EstimatorTypeEnum.MDN, version="5"),
            ModelCandidate(type=EstimatorTypeEnum.MDN, version="6"),
            ModelCandidate(type=EstimatorTypeEnum.MDN, version="7"),
            ModelCandidate(type=EstimatorTypeEnum.CVAE, version="0"),
            ModelCandidate(type=EstimatorTypeEnum.CVAE, version="1"),
        ],
        forward_estimator_type=EstimatorTypeEnum.COCO,
        target_objective=[410, 1400],
        distance_tolerance=0.02,
        n_samples=10,
    )

    # Initialize the handler with pre-built services
    handler = GenerateDecisionCommandHandler(
        workflow=DecisionGenerationWorkflow(),
        model_repository=FileSystemModelArtifactRepository(),
        data_repository=FileSystemDatasetRepository(),
        logger=CMDLogger(name="InterpolationCMDLogger"),
        visualizer=DecisionGenerationComparisonVisualizer(),
    )

    handler.execute(command)


if __name__ == "__main__":
    main()
