from typing import Any, Dict, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ....domain.common.interfaces.base_logger import BaseLogger
from ....domain.datasets.entities.processed_dataset import ProcessedDataset
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
        generated_data_repository: BaseDatasetRepository,
        processed_data_repository: BaseDatasetRepository,
        logger: BaseLogger,
    ):
        self._model_repository = model_repository
        self._processed_data_repo = processed_data_repository
        self._generated_data_repo = generated_data_repository
        self._logger = logger

    def execute(self, command: GenerateDecisionCommand) -> dict[str, np.ndarray]:
        """
        Generate candidates for the target.
        Returns dictionary with 'candidates' (raw) and 'candidates_norm'.
        """
        # 1. Load Resources
        inverse_estimator, forward_estimator, processed_data = self._load_context(
            command
        )

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

        # 5. Prepare Calibration Data
        calibration_data = self._prepare_calibration_data(
            processed_data, inverse_estimator
        )

        # 6. Visualize Results
        self._visualize_results(
            pareto_front=self._generated_data_repo.pareto.front,
            target_objective=target_y_raw.flatten(),
            predicted_objectives=predicted_objectives,
            generated_decisions=candidates_raw,
            calibration_data=calibration_data,
        )

        # 7. Return Results
        return {
            "decisions": candidates_raw,
            "decisions_norm": candidates_norm,
        }

    def _load_context(
        self, command: GenerateDecisionCommand
    ) -> Tuple[BaseEstimator, BaseEstimator, ProcessedDataset]:
        """Loads the necessary estimators and dataset."""
        inverse_estimator = self._load_estimator(
            command.inverse_estimator_type, direction="inverse"
        )
        forward_estimator = self._load_estimator(
            command.forward_estimator_type, direction="forward"
        )
        processed_data: ProcessedDataset = self._processed_data_repo.load(
            "dataset", variant="processed"
        )
        return inverse_estimator, forward_estimator, processed_data

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
        self, target_objective: list, processed_data: ProcessedDataset
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

    def _prepare_calibration_data(
        self, processed_data: ProcessedDataset, estimator: BaseEstimator
    ) -> Dict[str, np.ndarray]:
        """Prepares a subset of test data for calibration analysis."""
        n_calib = 100
        n_test = len(processed_data.objectives_test)
        indices = np.random.choice(n_test, size=min(n_calib, n_test), replace=False)

        test_objectives_subset_norm = processed_data.objectives_test[indices]
        test_decisions_subset_norm = processed_data.decisions_test[indices]

        # Denormalize for visualization
        test_objectives_subset_raw = (
            processed_data.objectives_normalizer.inverse_transform(
                test_objectives_subset_norm
            )
        )
        test_decisions_subset_raw = (
            processed_data.decisions_normalizer.inverse_transform(
                test_decisions_subset_norm
            )
        )

        # Generate samples for calibration (normalized inputs -> normalized outputs)
        calib_samples_norm = estimator.sample(test_objectives_subset_norm, n_samples=50)

        # Ensure 3D shape (N, S, D)
        if calib_samples_norm.ndim == 2:
            calib_samples_norm = calib_samples_norm[:, np.newaxis, :]

        return {
            "test_objectives_raw": test_objectives_subset_raw,
            "test_decisions_raw": test_decisions_subset_raw,
            "test_decisions_norm": test_decisions_subset_norm,
            "calib_samples_norm": calib_samples_norm,
        }

    def _visualize_results(
        self,
        pareto_front: np.ndarray,
        target_objective: np.ndarray,
        predicted_objectives: np.ndarray,
        generated_decisions: np.ndarray,
        calibration_data: Dict[str, np.ndarray],
    ):
        """Orchestrates the visualization of results."""
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=(
                "Objective Space",
                "Ghost Plot (Manifold)",
                "Calibration Curve",
            ),
            horizontal_spacing=0.1,
        )

        self._plot_objective_space(
            fig, pareto_front, predicted_objectives, target_objective, row=1, col=1
        )

        self._plot_ghost_manifold(
            fig,
            calibration_data["test_objectives_raw"],
            calibration_data["test_decisions_raw"],
            target_objective,
            generated_decisions,
            row=1,
            col=2,
        )

        self._plot_calibration_curve(
            fig,
            calibration_data["calib_samples_norm"],
            calibration_data["test_decisions_norm"],
            row=1,
            col=3,
        )

        fig.update_layout(
            title="Decision Generation & Validation Analysis",
            template="plotly_white",
            width=1500,
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

    def _plot_ghost_manifold(
        self,
        fig: go.Figure,
        test_objectives_raw: np.ndarray,
        test_decisions_raw: np.ndarray,
        target_objective: np.ndarray,
        generated_decisions: np.ndarray,
        row: int,
        col: int,
    ):
        """Plots the ghost plot (manifold check)."""
        # Background: Ground Truth Manifold (Test Subset)
        fig.add_trace(
            go.Scatter(
                x=test_objectives_raw[:, 0],
                y=test_decisions_raw[:, 0],
                mode="markers",
                name="Ground Truth",
                marker=dict(color="black", size=4, opacity=0.3),
                showlegend=True,
            ),
            row=row,
            col=col,
        )

        # Foreground: Model's Samples for Current Target
        target_y1 = target_objective[0]
        fig.add_trace(
            go.Scatter(
                x=np.full(len(generated_decisions), target_y1),
                y=generated_decisions[:, 0],
                mode="markers",
                name="Model Samples",
                marker=dict(color="blue", size=5, opacity=0.5),
                showlegend=True,
            ),
            row=row,
            col=col,
        )

        fig.update_xaxes(title_text="Target Y1", row=row, col=col)
        fig.update_yaxes(title_text="Decision X1", row=row, col=col)

    def _plot_calibration_curve(
        self,
        fig: go.Figure,
        calib_samples_norm: np.ndarray,
        test_decisions_norm: np.ndarray,
        row: int,
        col: int,
    ):
        """Plots the calibration curve (PIT CDF)."""
        n_test, n_samples, d_x = calib_samples_norm.shape
        pit_values = []

        for i in range(n_test):
            for d in range(d_x):
                true_val = test_decisions_norm[i, d]
                samples = calib_samples_norm[i, :, d]
                # PIT = P(sample <= true_val)
                pit = np.mean(samples <= true_val)
                pit_values.append(pit)

        pit_values = np.sort(pit_values)
        cdf_y = np.arange(1, len(pit_values) + 1) / len(pit_values)

        # Plot PIT CDF
        fig.add_trace(
            go.Scatter(
                x=pit_values,
                y=cdf_y,
                mode="lines",
                name="Calibration",
                line=dict(color="blue", width=2),
                showlegend=True,
            ),
            row=row,
            col=col,
        )

        # Plot Diagonal (Ideal)
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Ideal",
                line=dict(color="black", dash="dash"),
                showlegend=True,
            ),
            row=row,
            col=col,
        )

        fig.update_xaxes(
            title_text="Predicted Confidence", range=[0, 1], row=row, col=col
        )
        fig.update_yaxes(
            title_text="Observed Frequency", range=[0, 1], row=row, col=col
        )
