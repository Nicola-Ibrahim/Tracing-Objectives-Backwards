import click

from ...modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...modules.evaluation.application.use_cases.compare_inverse_model_candidates import (
    CompareInverseModelCandidatesCommand,
    CompareInverseModelCandidatesCommandHandler,
    InverseEstimatorCandidate,
)
from ...modules.evaluation.application.use_cases.compare_inverse_model_candidates.inverse_model_candidates_comparator import (
    InverseGeneratorComparator,
)
from ...modules.evaluation.infrastructure.visualization.decision_generation.visualizer import (
    DecisionGenerationComparisonVisualizer,
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
    command = CompareInverseModelCandidatesCommand(
        dataset_name="cocoex_f5",
        inverse_estimators=[
            InverseEstimatorCandidate(type=EstimatorTypeEnum.MDN, version=1),
            InverseEstimatorCandidate(type=EstimatorTypeEnum.MDN, version=2),
            InverseEstimatorCandidate(type=EstimatorTypeEnum.MDN, version=3),
            InverseEstimatorCandidate(type=EstimatorTypeEnum.MDN, version=4),
            InverseEstimatorCandidate(type=EstimatorTypeEnum.MDN, version=5),
            InverseEstimatorCandidate(type=EstimatorTypeEnum.MDN, version=6),
            InverseEstimatorCandidate(type=EstimatorTypeEnum.MDN, version=7),
            InverseEstimatorCandidate(type=EstimatorTypeEnum.INN, version=1),
            InverseEstimatorCandidate(type=EstimatorTypeEnum.CVAE, version=1),
            InverseEstimatorCandidate(type=EstimatorTypeEnum.CVAE, version=2),
            InverseEstimatorCandidate(type=EstimatorTypeEnum.CVAE, version=3),
        ],
        forward_estimator_type=EstimatorTypeEnum.COCO,
        target_objective=[410, 1400],
        distance_tolerance=0.02,
        n_samples=20,
        diversity_method="euclidean",
    )

    # Initialize the handler with pre-built services
    handler = CompareInverseModelCandidatesCommandHandler(
        comparator=InverseGeneratorComparator(),
        model_repository=FileSystemModelArtifactRepository(),
        data_repository=FileSystemDatasetRepository(),
        logger=CMDLogger(name="InterpolationCMDLogger"),
        visualizer=DecisionGenerationComparisonVisualizer(),
    )

    handler.execute(command)


if __name__ == "__main__":
    main()
