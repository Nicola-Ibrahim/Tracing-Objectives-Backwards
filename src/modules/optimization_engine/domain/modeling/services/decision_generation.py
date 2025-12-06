from typing import Any, Tuple

import numpy as np
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
        """
        model_names = list(results_map.keys())
        n_models = len(model_names)

        fig = make_subplots(
            rows=1,
            cols=n_models,
            subplot_titles=[f"{name} Predictions" for name in model_names],
        )

        colors = ["blue", "green", "purple", "orange", "cyan"]

        for idx, (model_name, res) in enumerate(results_map.items()):
            col = idx + 1
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
                row=1,
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
                row=1,
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
                row=1,
                col=col,
            )

            # Update Axes
            fig.update_xaxes(title_text="Objective 1", row=1, col=col)
            fig.update_yaxes(title_text="Objective 2", row=1, col=col)

        fig.update_layout(
            title="Decision Generation Comparison (Separate Views)",
            template="plotly_white",
            width=600 * n_models,
            height=600,
        )

        return fig
