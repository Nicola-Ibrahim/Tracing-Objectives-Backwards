import click

from ...modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...modules.evaluation.application.use_cases import (
    CompareInverseModelCandidatesParams,
    CompareInverseModelCandidatesService,
    InverseEstimatorCandidate,
)
from ...modules.evaluation.application.use_cases.inverse_model_candidates_comparator import (
    InverseModelsCandidatesComparator,
)
from ...modules.evaluation.infrastructure.visualization.decision_generation.visualizer import (
    DecisionGenerationComparisonVisualizer,
)
from ...modules.modeling.domain.enums.estimator_type import EstimatorTypeEnum
from ...modules.modeling.infrastructure.repositories.trained_pipeline_repo import (
    FileSystemTrainedPipelineRepository,
)
from ...modules.shared.infrastructure.loggers.cmd_logger import CMDLogger


@click.command(help="Generate decision candidates for a target objective.")
def main():
    """
    Main function to generate a decision using parameters.
    """

    # Create the command object using the provided estimator and hardcoded target
    params = CompareInverseModelCandidatesParams(
        dataset_name="cocoex_f5",
        inverse_estimators=[
            InverseEstimatorCandidate(type=EstimatorTypeEnum.MDN, version=8),
            InverseEstimatorCandidate(type=EstimatorTypeEnum.MDN, version=10),
            InverseEstimatorCandidate(type=EstimatorTypeEnum.MDN, version=12),
            InverseEstimatorCandidate(type=EstimatorTypeEnum.MDN, version=13),
            InverseEstimatorCandidate(type=EstimatorTypeEnum.CVAE, version=1),
            InverseEstimatorCandidate(type=EstimatorTypeEnum.CVAE, version=2),
            InverseEstimatorCandidate(type=EstimatorTypeEnum.INN, version=1),
            InverseEstimatorCandidate(type=EstimatorTypeEnum.INN, version=5),
        ],
        forward_estimator_type=EstimatorTypeEnum.COCO,
        target_objective=[408, 1300],
        n_samples=20,
    )

    # Initialize the handler with pre-built services
    service = CompareInverseModelCandidatesService(
        comparator=InverseModelsCandidatesComparator(),
        model_repository=FileSystemTrainedPipelineRepository(),
        data_repository=FileSystemDatasetRepository(),
        logger=CMDLogger(name="InterpolationCMDLogger"),
        visualizer=DecisionGenerationComparisonVisualizer(),
    )

    service.execute(params)


if __name__ == "__main__":
    main()
