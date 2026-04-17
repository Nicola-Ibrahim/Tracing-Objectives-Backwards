import click

from ...modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...modules.inverse.application.train_inverse_mapping_engine import (
    TrainInverseMappingEngineParams,
    TrainInverseMappingEngineService,
)
from ...modules.inverse.infrastructure.repositories.inverse_mapping_engine_repo import (
    FileSystemInverseMappingEngineRepository,
)
from ...modules.inverse.infrastructure.solvers.factory import SolversFactory
from ...modules.modeling.infrastructure.factories.transformer import TransformerFactory
from ...modules.shared.infrastructure.loggers.cmd_logger import CMDLogger


@click.command()
@click.option("--dataset_name", required=True, type=str, help="Name of the dataset")
@click.option("--solver_type", default="GBPI", help="Type of inverse solver to use")
@click.option(
    "--split_ratio", default=0.2, type=float, help="Ratio of data used for testing"
)
@click.option("--random_state", default=42, type=int, help="Random seed for splitting")
def main(dataset_name: str, solver_type: str, split_ratio: float, random_state: int):
    """
    Offline step: Prepare the inverse mapping engine for a dataset.
    """
    logger = CMDLogger(name="PrepareEngineLogger")

    params = TrainInverseMappingEngineParams(
        dataset_name=dataset_name,
        split_ratio=split_ratio,
        random_state=random_state,
        transforms=[
            {"type": "min_max", "target": "decisions"},
            {"type": "min_max", "target": "objectives"},
        ],
        solver={
            "type": solver_type,
            "params": {
                "n_neighbors": 5,
                "trust_radius": 0.05,
                "concentration_factor": 10.0,
            },
        },
    )

    service = TrainInverseMappingEngineService(
        dataset_repository=FileSystemDatasetRepository(),
        inverse_mapping_engine_repository=FileSystemInverseMappingEngineRepository(),
        logger=logger,
        transformer_factory=TransformerFactory(),
        solvers_factory=SolversFactory(),
    )
    service.execute(params)


if __name__ == "__main__":
    main()
