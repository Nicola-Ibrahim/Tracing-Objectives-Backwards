import click

from ...modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...modules.evaluation.application.compare_candidates import (
    CompareInverseModelCandidatesParams,
    CompareInverseModelCandidatesService,
)
from ...modules.evaluation.application.diagnose_engines import EngineCandidate
from ...modules.evaluation.application.inverse_model_candidates_comparator import (
    InverseModelsCandidatesComparator,
)
from ...modules.evaluation.infrastructure.visualization import (
    decision_generation as dg,
)
from ...modules.inverse.infrastructure.repositories.inverse_mapping_engine_repo import (
    FileSystemInverseMappingEngineRepository,
)
from ...modules.shared.infrastructure.loggers.cmd_logger import CMDLogger


@click.command(help="Generate decision candidates for a target objective.")
def main():
    """
    Main function to generate a decision using parameters.
    """

    params = CompareInverseModelCandidatesParams(
        dataset_name="cocoex_f5",
        inverse_engines=[
            EngineCandidate(solver_type="GBPI", version=1),
        ],
        target_objective=[408, 1300],
        n_samples=20,
    )

    # Initialize the handler with pre-built services
    service = CompareInverseModelCandidatesService(
        comparator=InverseModelsCandidatesComparator(),
        engine_repository=FileSystemInverseMappingEngineRepository(),
        data_repository=FileSystemDatasetRepository(),
        logger=CMDLogger(name="InterpolationCMDLogger"),
        visualizer=dg.visualizer.DecisionGenerationComparisonVisualizer(),
    )

    service.execute(params)


if __name__ == "__main__":
    main()
