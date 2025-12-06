from typing import Any, Tuple

import numpy as np

from ....domain.common.interfaces.base_logger import BaseLogger
from ....domain.datasets.entities.dataset import Dataset
from ....domain.datasets.entities.processed_data import ProcessedData
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ....domain.modeling.enums.estimator_type import EstimatorTypeEnum
from ....domain.modeling.interfaces.base_estimator import BaseEstimator
from ....domain.modeling.interfaces.base_repository import BaseModelArtifactRepository
from ....domain.modeling.services.decision_generation import DecisionGenerator
from .command import GenerateDecisionCommand


class GenerateDecisionCommandHandler:
    """
    Generate Design Candidates (X) for a requested Objective (Y).

    Unified Handler for Multiple Models:
    - Uses DecisionGenerator domain service.
    - Compares generation capability across multiple inverse models (MDN, CVAE, etc).
    """

    def __init__(
        self,
        model_repository: BaseModelArtifactRepository,
        data_repository: BaseDatasetRepository,
        logger: BaseLogger,
        generator_service: DecisionGenerator,
    ):
        self._model_repository = model_repository
        self._data_repository = data_repository
        self._logger = logger
        self._generator_service = generator_service

    def execute(self, command: GenerateDecisionCommand) -> dict[str, Any]:
        """
        Generate candidates for the target using multiple models.
        Returns dictionary with results for each model.
        """
        # 1. Load Resources (Forward Model & Dataset)
        forward_estimator = self._load_estimator(
            command.forward_estimator_type, direction="forward"
        )

        dataset: Dataset = self._data_repository.load("dataset")
        processed_data = dataset.processed
        if not processed_data:
            raise ValueError(f"Dataset '{dataset.name}' has no processed data.")

        # 2. Prepare Target
        target_objective_raw, target_objective_norm = self._prepare_target(
            command.target_objective, processed_data
        )

        self._logger.log_info(
            f"Target Objective (Raw): {target_objective_raw.tolist()}"
        )

        # 3. Iterate over Inverse Model Candidates & Generate
        results_map = {}

        for candidate in command.inverse_candidates:
            inverse_type = candidate.type.value
            version_int = candidate.version

            try:
                # Determine display name
                if version_int is not None:
                    display_name = f"{inverse_type} (v{version_int})"
                    self._logger.log_info(
                        f"Generating decisions with {display_name}..."
                    )

                    # Load specific version
                    all_versions = self._model_repository.get_all_versions(
                        estimator_type=inverse_type,
                        mapping_direction="inverse",
                    )
                    target_artifact = next(
                        (a for a in all_versions if a.version == version_int), None
                    )

                    if not target_artifact:
                        raise ValueError(
                            f"Version {version_int} not found for {inverse_type}"
                        )

                    inverse_estimator = target_artifact.estimator
                else:
                    display_name = f"{inverse_type} (Latest)"
                    self._logger.log_info(
                        f"Generating decisions with {display_name}..."
                    )

                    # Load latest version
                    inverse_artifact = self._model_repository.get_latest_version(
                        estimator_type=inverse_type,
                        mapping_direction="inverse",
                    )
                    inverse_estimator = inverse_artifact.estimator

                # Generate Candidates via Service
                candidates_raw, candidates_norm = self._generator_service.generate(
                    estimator=inverse_estimator,
                    target_objective_norm=target_objective_norm,
                    n_samples=command.n_samples,
                    decisions_normalizer=processed_data.decisions_normalizer,
                )

                # Predict Outcomes via Forward Model
                predicted_objectives = self._generator_service.predict_outcomes(
                    forward_estimator=forward_estimator,
                    candidates_raw=candidates_raw,
                )

                # Store results
                results_map[display_name] = {
                    "decisions": candidates_raw,
                    "predicted_objectives": predicted_objectives,
                }

            except Exception as e:
                self._logger.log_error(f"Failed to generate with {inverse_type}: {e}")
                continue
        # 4. Visualization (Comparison)
        if results_map:
            self._logger.log_info("Generating comparison visualization...")
            fig = self._generator_service.compare_generators(
                results_map=results_map,
                pareto_front=dataset.pareto.front,
                target_objective=target_objective_raw.flatten(),
            )
            try:
                fig.show()
            except Exception as e:
                self._logger.log_warning(f"Could not display plot: {e}")

        # 5. Return Results
        return results_map

    def _load_estimator(
        self, estimator_type: EstimatorTypeEnum, direction: str = "inverse"
    ) -> BaseEstimator:
        """Helper to load an estimator from the repository."""
        artifact = self._model_repository.get_latest_version(
            estimator_type=estimator_type.value,
            mapping_direction=direction,
        )
        return artifact.estimator

    def _prepare_target(
        self, target_objective: list, processed_data: ProcessedData
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepares raw and normalized target objectives."""
        target_objective_raw = np.array(target_objective, dtype=float).reshape(1, -1)
        target_objective_norm = processed_data.objectives_normalizer.transform(
            target_objective_raw
        )
        return target_objective_raw, target_objective_norm
