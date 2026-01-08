from typing import Any

import numpy as np

from ....domain.common.interfaces.base_logger import BaseLogger
from ....domain.common.interfaces.base_visualizer import BaseVisualizer
from ....domain.datasets.entities.dataset import Dataset
from ....domain.datasets.entities.processed_data import ProcessedData
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ....domain.modeling.interfaces.base_estimator import BaseEstimator
from ....domain.modeling.interfaces.base_repository import BaseModelArtifactRepository
from ....workflows.decision_generation_workflow import DecisionGenerationWorkflow
from .command import GenerateDecisionCommand, InverseEstimatorCandidate


class GenerateDecisionCommandHandler:
    """
    Generate Design Candidates (X) for a requested Objective (Y).

    Unified Handler for Multiple Models:
    - Uses a dedicated decision-generation workflow.
    - Compares generation capability across multiple inverse models (MDN, CVAE, etc).
    """

    def __init__(
        self,
        workflow: DecisionGenerationWorkflow,
        model_repository: BaseModelArtifactRepository,
        data_repository: BaseDatasetRepository,
        logger: BaseLogger,
        visualizer: BaseVisualizer | None = None,
    ):
        self._workflow = workflow
        self._model_repository = model_repository
        self._data_repository = data_repository
        self._logger = logger
        self._visualizer = visualizer

    def execute(self, command: GenerateDecisionCommand) -> dict[str, Any]:
        """
        Coordinates the generation of decision candidates using multiple inverse models.
        """
        self._logger.log_info(
            f"Starting decision generation on '{command.dataset_name}'. "
            f"Candidates: {[(c.type.value, c.version) for c in command.inverse_estimators]}, "
            f"Forward: {command.forward_estimator_type.value}"
        )

        # 1. Load context: dataset and processed data
        dataset: Dataset = self._data_repository.load(command.dataset_name)
        if not dataset.processed:
            raise ValueError(f"Dataset '{dataset.name}' has no processed data.")

        # 2. Prepare target objective (raw + normalized)
        target_objective_raw, target_objective_norm = self._prepare_target(
            command.target_objective, dataset.processed
        )
        self._logger.log_info(
            f"Target Objective (Raw): {target_objective_raw.tolist()}"
        )

        # 3. Initialize inverse estimators using private helper
        inverse_estimators = self._initialize_estimators(
            candidates=command.inverse_estimators, dataset_name=command.dataset_name
        )

        # 4. Load forward estimator using private helper
        forward_estimator = self._load_forward_estimator(
            estimator_type=command.forward_estimator_type.value,
            dataset_name=command.dataset_name,
        )

        # 5. Run generation workflow
        # TODO: Edit inverse_estimators to use dictionary instead of tuple
        workflow_output: dict[str, object] = self._workflow.run(
            inverse_estimators=list(inverse_estimators.items()),
            pareto_front=dataset.pareto.front,
            target_objective_raw=target_objective_raw,
            target_objective_norm=target_objective_norm,
            decisions_normalizer=dataset.processed.decisions_normalizer,
            forward_estimator=forward_estimator,
            n_samples=command.n_samples,
            distance_tolerance=command.distance_tolerance,
        )

        # 6. Build visualization payload and generate plots
        results_map = workflow_output["results_map"]
        generator_runs = workflow_output["generator_runs"]

        if results_map and self._visualizer:
            self._logger.log_info("Generating comparison visualization...")
            visualization_data = self._build_visualization_payload(
                dataset=dataset,
                target_objective=target_objective_raw,
                generator_runs=generator_runs,
            )
            self._visualizer.plot(visualization_data)

        return results_map

    def _initialize_estimators(
        self, candidates: list[InverseEstimatorCandidate], dataset_name: str
    ) -> dict[str, BaseEstimator]:
        """
        Initializes inverse estimator instances from the repository.
        """
        inverse_estimators: dict[str, BaseEstimator] = {}

        for candidate in candidates:
            inverse_type = candidate.type.value
            version = candidate.version
            display_name = (
                f"{inverse_type} (v{version})"
                if version
                else f"{inverse_type} (latest)"
            )

            # Resolve from repository
            artifact = self._model_repository.get_version_by_number(
                estimator_type=inverse_type,
                version=version,
                mapping_direction="inverse",
                dataset_name=dataset_name,
            )

            inverse_estimators[display_name] = artifact.estimator

        return inverse_estimators

    def _load_forward_estimator(
        self, estimator_type: str, dataset_name: str
    ) -> BaseEstimator:
        """
        Loads the forward estimator from the repository.
        """
        self._logger.log_info(f"Loading forward estimator: {estimator_type}...")
        artifact = self._model_repository.get_latest_version(
            estimator_type=estimator_type,
            mapping_direction="forward",
            dataset_name=dataset_name,
        )
        return artifact.estimator

    def _build_visualization_payload(
        self,
        dataset: Dataset,
        target_objective: np.ndarray,
        generator_runs: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Builds the visualization payload for the visualizer.
        """
        return {
            "dataset_name": dataset.name,
            "pareto_front": dataset.pareto.front,
            "target_objective": target_objective,
            "generators": generator_runs,
        }

    def _prepare_target(
        self, target_objective: list, processed_data: ProcessedData
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepares raw and normalized target objectives.
        """
        target_objective_raw = np.array(target_objective, dtype=float).reshape(1, -1)
        target_objective_norm = processed_data.objectives_normalizer.transform(
            target_objective_raw
        )
        return target_objective_raw, target_objective_norm
