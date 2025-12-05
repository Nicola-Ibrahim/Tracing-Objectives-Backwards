from typing import Any, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ....domain.common.interfaces.base_logger import BaseLogger
from ....domain.datasets.entities.dataset import Dataset
from ....domain.datasets.entities.processed_data import ProcessedData
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ....domain.modeling.enums.estimator_type import EstimatorTypeEnum
from ....domain.modeling.interfaces.base_estimator import BaseEstimator
from ....domain.modeling.interfaces.base_repository import BaseModelArtifactRepository
from .command import GenerateDecisionCommand


class GenerateDecisionCommandHandler:
    """
    Generate Design Candidates (X) for a requested Objective (Y).

    Unified Handler for MDN and CVAE:
    - Removes external Feasibility/Validation services (redundant).
    - Adapts to model capabilities:
        - If MDN: Uses analytic `predict_topk` to find exact modes.
        - If CVAE: Uses `sample` to generate a cloud of options.
    """

    def __init__(
        self,
        model_repository: BaseModelArtifactRepository,
        data_repository: BaseDatasetRepository,
        logger: BaseLogger,
    ):
        self._model_repository = model_repository
        self._data_repository = data_repository
        self._logger = logger

    def execute(self, command: GenerateDecisionCommand) -> dict[str, np.ndarray]:
        """
        Generate candidates for the target.
        Returns dictionary with 'candidates' (raw) and 'candidates_norm'.
        """
        # 1. Load Resources
        inverse_estimator, forward_estimator, dataset = self._load_context(command)
        processed_data = dataset.processed
        if not processed_data:
            raise ValueError(f"Dataset '{dataset.name}' has no processed data.")

        # 2. Prepare Target
        target_y_raw, target_y_norm = self._prepare_target(
            command.target_objective, processed_data
        )

        self._logger.log_info(f"Target Objective (Raw): {target_y_raw.tolist()}")

        # 3. Generate Candidates
        candidates_raw, candidates_norm = self._generate_decisions(
            inverse_estimator,
            target_y_norm,
            command.n_samples,
            processed_data.decisions_normalizer,
        )

        # 4. Predict Outcomes (Verification)
        predicted_objectives = self._predict_outcomes(forward_estimator, candidates_raw)

        # 5. Visualize Results
        self._visualize_results(
            pareto_front=dataset.pareto.front,
            target_objective=target_y_raw.flatten(),
            predicted_objectives=predicted_objectives,
            generated_decisions=candidates_raw,
        )

        # 6. Return Results
        return {
            "decisions": candidates_raw,
            "decisions_norm": candidates_norm,
        }

    def _load_context(
        self, command: GenerateDecisionCommand
    ) -> Tuple[BaseEstimator, BaseEstimator, Dataset]:
        """Loads the necessary estimators and dataset."""
        inverse_estimator = self._load_estimator(
            command.inverse_estimator_type, direction="inverse"
        )
        forward_estimator = self._load_estimator(
            command.forward_estimator_type, direction="forward"
        )
        dataset: Dataset = self._data_repository.load("dataset")

        return inverse_estimator, forward_estimator, dataset

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
        target_y_raw = np.array(target_objective, dtype=float).reshape(1, -1)
        target_y_norm = processed_data.objectives_normalizer.transform(target_y_raw)
        return target_y_raw, target_y_norm

    def _generate_decisions(
        self,
        estimator: BaseEstimator,
        target_y_norm: np.ndarray,
        n_samples: int,
        decisions_normalizer: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates decision candidates and handles denormalization."""
        candidates_norm = estimator.sample(target_y_norm, n_samples=n_samples)

        # Ensure 2D shape for inverse_transform: (N_samples, x_dim)
        if candidates_norm.ndim == 3:
            candidates_norm = candidates_norm.reshape(-1, candidates_norm.shape[-1])

        candidates_raw = decisions_normalizer.inverse_transform(candidates_norm)
        return candidates_raw, candidates_norm

    def _predict_outcomes(
        self, forward_estimator: BaseEstimator, candidates_raw: np.ndarray
    ) -> np.ndarray:
        """Predicts objectives for the generated candidates using the forward model."""
        return forward_estimator.predict(candidates_raw)

    def _visualize_results(
        self,
        pareto_front: np.ndarray,
        target_objective: np.ndarray,
        predicted_objectives: np.ndarray,
        generated_decisions: np.ndarray,
    ):
        """Orchestrates the visualization of results."""
        # Single plot for Objective Space
        fig = make_subplots(
            rows=1,
            cols=1,
            subplot_titles=("Objective Space",),
        )

        self._plot_objective_space(
            fig, pareto_front, predicted_objectives, target_objective, row=1, col=1
        )

        fig.update_layout(
            title="Decision Generation Analysis",
            template="plotly_white",
            width=800,
            height=600,
            showlegend=True,
        )

        fig.show()

    def _plot_objective_space(
        self,
        fig: go.Figure,
        pareto_front: np.ndarray,
        predicted_objectives: np.ndarray,
        target_objective: np.ndarray,
        row: int,
        col: int,
    ):
        """Plots the objective space analysis."""
        # Background: Pareto Front
        fig.add_trace(
            go.Scatter(
                x=pareto_front[:, 0],
                y=pareto_front[:, 1],
                mode="markers",
                name="Pareto Front",
                marker=dict(color="lightgray", size=5, opacity=0.5),
                showlegend=True,
            ),
            row=row,
            col=col,
        )

        # Foreground: Predicted Cloud
        fig.add_trace(
            go.Scatter(
                x=predicted_objectives[:, 0],
                y=predicted_objectives[:, 1],
                mode="markers",
                name="Predicted Outcomes",
                marker=dict(color="blue", size=6, opacity=0.7),
                showlegend=True,
            ),
            row=row,
            col=col,
        )

        # Target: Star
        fig.add_trace(
            go.Scatter(
                x=[target_objective[0]],
                y=[target_objective[1]],
                mode="markers",
                name="Target Objective",
                marker=dict(
                    color="red",
                    symbol="star",
                    size=15,
                    line=dict(width=2, color="black"),
                ),
                showlegend=True,
            ),
            row=row,
            col=col,
        )

        fig.update_xaxes(title_text="Objective 1", row=row, col=col)
        fig.update_yaxes(title_text="Objective 2", row=row, col=col)
