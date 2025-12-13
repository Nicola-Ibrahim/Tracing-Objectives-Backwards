import click

from ...application.assuring.select_inverse_model.command import ModelCandidate
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
from ...infrastructure.loggers.cmd_logger import CMDLogger
from ...infrastructure.modeling.repositories.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)
from ...infrastructure.modeling.visualizers.decision_generation import (
    DecisionGenerationComparisonVisualizer,
)
from ...workflows.decision_generation_workflow import DecisionGenerationWorkflow


@click.command(help="Generate decision candidates for a target objective.")
def main():
    """
    Main function to generate a decision using parameters.
    """

    # Create the command object using the provided estimator and hardcoded target
    command = GenerateDecisionCommand(
        inverse_candidates=[
            ModelCandidate(type=EstimatorTypeEnum.MDN, version="1"),
            ModelCandidate(type=EstimatorTypeEnum.CVAE, version="2"),
            ModelCandidate(type=EstimatorTypeEnum.MDN, version="3"),
            ModelCandidate(type=EstimatorTypeEnum.CVAE, version="1"),
        ],
        forward_estimator_type=EstimatorTypeEnum.COCO,
        target_objective=[408, 1282],
        distance_tolerance=0.02,
        n_samples=50,
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
