from typing import Any, Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..interfaces.base_estimator import BaseEstimator


class DecisionGenerator:
    """
    Domain service for generating and analyzing design candidates (decisions)
    using inverse models.
    """

    def generate(
        self,
        estimator: BaseEstimator,
        target_objective_norm: np.ndarray,
        n_samples: int,
        decisions_normalizer: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates decision candidates for a normalized target objective.
        Returns (raw_decisions, norm_decisions).
        """
        # Sample from estimator (normalized space)
        candidates_norm = estimator.sample(target_objective_norm, n_samples=n_samples)

        # Ensure correct shape (N_samples, x_dim)
        if candidates_norm.ndim == 3:
            # Flatten/Squeeze if the estimator returns (N_targets, N_samples, Dim)
            # Since target is usually 1 here, we take index 0 or reshape
            candidates_norm = candidates_norm.reshape(-1, candidates_norm.shape[-1])

        # Denormalize
        candidates_raw = decisions_normalizer.inverse_transform(candidates_norm)

        return candidates_raw, candidates_norm

    def predict_outcomes(
        self, forward_estimator: BaseEstimator, candidates_raw: np.ndarray
    ) -> np.ndarray:
        """
        Predicts objective outcomes for raw decision candidates using a forward model.
        Returns predicted_objectives.
        """
        return forward_estimator.predict(candidates_raw)

    def compare_generators(
        self,
        results_map: dict[
            str, dict
        ],  # {model_name: {'decisions': np.ndarray, 'predicted_objectives': np.ndarray}}
        pareto_front: np.ndarray,
        target_objective: np.ndarray,
    ) -> go.Figure:
        """
        Visualizes the generated solutions from multiple models in separate subplots.
        3 plots per row, stacked vertically.
        """
        model_names = list(results_map.keys())
        n_models = len(model_names)

        # Calculate rows and columns (3 per row)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[f"{name} Predictions" for name in model_names],
        )

        # Generate unique colors dynamically based on number of models
        if n_models <= 10:
            colors = px.colors.qualitative.Plotly
        elif n_models <= 24:
            colors = px.colors.qualitative.Dark24
        else:
            # Fall back to cycling through a large palette
            colors = px.colors.qualitative.Dark24 + px.colors.qualitative.Plotly

        for idx, (model_name, res) in enumerate(results_map.items()):
            row = (idx // n_cols) + 1
            col = (idx % n_cols) + 1
            predicted = res["predicted_objectives"]
            color = colors[idx % len(colors)]

            # 1. Background: Pareto Front
            fig.add_trace(
                go.Scatter(
                    x=pareto_front[:, 0],
                    y=pareto_front[:, 1],
                    mode="markers",
                    name="Pareto Front",
                    marker=dict(color="lightgray", size=5, opacity=0.3),
                    showlegend=(idx == 0),  # Only show legend once
                    legendgroup="pareto",
                ),
                row=row,
                col=col,
            )

            # 2. Target Objective (Star)
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
                    showlegend=(idx == 0),  # Only show legend once
                    legendgroup="target",
                ),
                row=row,
                col=col,
            )

            # 3. Model Predictions
            fig.add_trace(
                go.Scatter(
                    x=predicted[:, 0],
                    y=predicted[:, 1],
                    mode="markers",
                    name=f"{model_name}",
                    marker=dict(color=color, size=6, opacity=0.7),
                    showlegend=True,
                ),
                row=row,
                col=col,
            )

            # Update Axes
            fig.update_xaxes(title_text="Objective 1", row=row, col=col)
            fig.update_yaxes(title_text="Objective 2", row=row, col=col)

        fig.update_layout(
            title="Decision Generation Comparison (Grid View)",
            template="plotly_white",
            width=1800,  # Fixed width for 3 columns
            height=600 * n_rows,  # Scale height based on rows
        )

        return fig
