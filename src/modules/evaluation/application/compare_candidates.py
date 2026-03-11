from typing import Any

from pydantic import BaseModel, Field

from ...dataset.domain.entities.dataset import Dataset
from ...dataset.domain.interfaces.base_repository import BaseDatasetRepository
from ...inverse.domain.entities.inverse_mapping_engine import (
    InverseMappingEngine,
)
from ...inverse.domain.interfaces.base_inverse_mapping_engine_repository import (
    BaseInverseMappingEngineRepository,
)
from ...shared.domain.interfaces.base_logger import BaseLogger
from ..domain.interfaces.base_visualizer import BaseVisualizer
from .inverse_model_candidates_comparator import InverseModelsCandidatesComparator


from .diagnose_engines import EngineCandidate


class CompareInverseModelCandidatesParams(BaseModel):
    dataset_name: str = Field(
        ...,
        description="Dataset identifier to use for comparison.",
    )

    inverse_engines: list[EngineCandidate] = Field(
        ...,
        description="list of inverse engine candidates to use for generation",
    )

    target_objective: list[float] = Field(
        ...,
        description="Target point in objective space (y1, y2, ...)",
    )
    n_samples: int = Field(
        ...,
        description="Number of samples to draw per inverse engine.",
    )


class CompareInverseModelCandidatesService:
    """
    Generate Design Candidates (X) for a requested Objective (Y).
    Unified Handler for Multiple Engines.
    """

    def __init__(
        self,
        engine_repository: BaseInverseMappingEngineRepository,
        data_repository: BaseDatasetRepository,
        logger: BaseLogger,
        visualizer: BaseVisualizer,
    ):
        self._engine_repository = engine_repository
        self._data_repository = data_repository
        self._logger = logger
        self._visualizer = visualizer

    def execute(self, params: CompareInverseModelCandidatesParams) -> dict[str, Any]:
        """
        Coordinates the generation of decision candidates using multiple inverse engines.
        """
        self._logger.log_info(
            f"Starting decision generation comparison on '{params.dataset_name}'."
        )

        dataset: Dataset = self._data_repository.load(params.dataset_name)

        engines = self._initialize_engines(
            candidates=params.inverse_engines, dataset_name=params.dataset_name
        )

        workflow_output = InverseModelsCandidatesComparator.compare(
            engines=engines,
            target_objective=params.target_objective,
            n_samples=params.n_samples,
        )

        results_map = workflow_output["results_map"]
        generator_runs = workflow_output["generator_runs"]

        self._logger.log_info("Generating comparison visualization...")
        self._visualizer.plot(
            data={
                "dataset_name": dataset.name,
                "pareto_front": dataset.pareto.front if dataset.pareto else [],
                "pareto_set": dataset.pareto.set if dataset.pareto else [],
                "target_objective": params.target_objective,
                "generators": generator_runs,
            }
        )

        return results_map

    def _initialize_engines(
        self, candidates: list[EngineCandidate], dataset_name: str
    ) -> dict[str, InverseMappingEngine]:
        """
        Initializes inverse engines from the repository.
        """
        engines: dict[str, InverseMappingEngine] = {}

        for candidate in candidates:
            version = candidate.version
            display_name = (
                f"{candidate.solver_type} (v{version})"
                if version
                else f"{candidate.solver_type} (latest)"
            )

            engine = self._engine_repository.load(
                dataset_name=dataset_name,
                solver_type=candidate.solver_type,
                version=version,
            )

            engines[display_name] = engine

        return engines
