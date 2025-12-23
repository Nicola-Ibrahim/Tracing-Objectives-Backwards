from typing import Any, Tuple, cast

import numpy as np

from ....domain.common.interfaces.base_logger import BaseLogger
from ....domain.common.interfaces.base_visualizer import BaseVisualizer
from ....domain.datasets.entities.dataset import Dataset
from ....domain.datasets.entities.processed_data import ProcessedData
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ....domain.modeling.interfaces.base_estimator import BaseEstimator
from ....domain.modeling.interfaces.base_repository import BaseModelArtifactRepository
from ....workflows.decision_generation_workflow import DecisionGenerationWorkflow
from .command import GenerateDecisionCommand


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
        Generate candidates for the target using multiple models.
        Returns dictionary with results for each model.
        """
        # 1) Load dataset + processed artifacts (normalizers, train/test splits, pareto front).
        dataset_name = next(
            iter({c.dataset_name or "dataset" for c in command.inverse_estimators})
        )

        dataset: Dataset = self._data_repository.load(dataset_name)
        processed_data = dataset.processed
        if not processed_data:
            raise ValueError(f"Dataset '{dataset.name}' has no processed data.")

        # 2) Prepare target objective:
        #    - raw: in original objective units (used for selection + plotting)
        #    - norm: in normalized objective space (used as inverse-estimator input)
        target_objective_raw, target_objective_norm = self._prepare_target(
            command.target_objective, processed_data
        )

        self._logger.log_info(
            f"Target Objective (Raw): {target_objective_raw.tolist()}"
        )

        # 3) Resolve inverse estimators requested by the command (type + optional version).
        #    Version lookup and artifact reading are repository responsibilities.
        inverse_estimators = self._model_repository.get_estimators(
            mapping_direction="inverse",
            requested=[(c.type.value, c.version) for c in command.inverse_estimators],
            dataset_name=dataset_name,
        )

        # 4) Resolve the forward estimator used for:
        #    - predicting objective outcomes for generated candidates
        #    - ranking/selecting the best candidate per inverse estimator
        self._logger.log_info(
            f"Loading forward estimator: {command.forward_estimator_type.value}..."
        )
        forward_artifact = self._model_repository.get_latest_version(
            estimator_type=command.forward_estimator_type.value,
            mapping_direction="forward",
            dataset_name=dataset_name,
        )
        forward_estimator: BaseEstimator = forward_artifact.estimator

        # 5) Run generation workflow:
        #    - generates candidates per inverse estimator
        #    - predicts objectives via forward_estimator
        #    - optionally applies validators and selects the best candidate
        workflow_output: dict[str, object] = self._workflow.run(
            inverse_estimators=inverse_estimators,
            pareto_front=dataset.pareto.front,
            target_objective_raw=target_objective_raw,
            target_objective_norm=target_objective_norm,
            decisions_normalizer=processed_data.decisions_normalizer,
            forward_estimator=forward_estimator,
            n_samples=command.n_samples,
            distance_tolerance=command.distance_tolerance,
        )

        # 6) Collect results and build a visualization payload in the handler.
        #    The visualizer expects a plain dict with pareto_front/target/generators.
        results_map = cast(dict[str, dict], workflow_output["results_map"])
        generator_runs = cast(
            list[dict[str, object]], workflow_output["generator_runs"]
        )
        visualization_data = {
            "dataset_name": dataset.name,
            "pareto_front": dataset.pareto.front,
            "target_objective": target_objective_raw,
            "generators": generator_runs,
        }

        # 7) Optional visualization: show how each estimator sampled (pre-validation) and
        #    where the target lies relative to the pareto front.
        if results_map and self._visualizer:
            self._logger.log_info("Generating comparison visualization...")
            try:
                self._visualizer.plot(visualization_data)
            except Exception as e:
                self._logger.log_warning(f"Could not display plot: {e}")
        elif results_map:
            self._logger.log_info("Visualization skipped (no visualizer provided).")

        # 6. Return Results
        return results_map

    def _prepare_target(
        self, target_objective: list, processed_data: ProcessedData
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepares raw and normalized target objectives."""
        target_objective_raw = np.array(target_objective, dtype=float).reshape(1, -1)
        target_objective_norm = processed_data.objectives_normalizer.transform(
            target_objective_raw
        )
        return target_objective_raw, target_objective_norm
