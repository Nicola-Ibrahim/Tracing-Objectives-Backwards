from typing import Any

import numpy as np
import plotly.graph_objects as go


def add_accuracy_ecdf_plot(
    fig: go.Figure,
    row: int,
    col: int,
    results_map: dict[str, dict[str, Any]],
    color_map: dict[str, str],
) -> None:
    """
    Adds an Empirical Cumulative Distribution Function (ECDF) plot of the
    best-shot discrepancy scores for each model.
    """
    for model_name, res in results_map.items():
        # 1. Fetch best-shot scores
        # We handle both the new naming and fallback if needed
        accuracy = res.get("accuracy", {})
        best_shot = accuracy.get("best_shot_scores")

        if best_shot is None:
            # Fallback to discrepancy_scores if best_shot is missing
            discrepancy = accuracy.get("discrepancy_scores")
            if discrepancy is not None:
                best_shot = np.min(discrepancy, axis=1)

        if best_shot is None:
            continue

        # 2. Extract metadata components for the label
        meta = res.get("metadata", {})
        scale_method = meta.get("scale_method", "unknown")
        n_samples = meta.get("num_samples", "?")

        # Label format: Type v1 (scale, K=10)
        label = f"{model_name} ({scale_method}, K={n_samples})"

        # 3. Construct ECDF
        x_sorted = np.sort(best_shot)
        y_ecdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)

        color = color_map.get(model_name, "gray")

        # 4. Add ECDF trace
        fig.add_trace(
            go.Scatter(
                x=x_sorted,
                y=y_ecdf,
                mode="lines",
                name=label,
                line=dict(color=color, width=2.5, shape="hv"),
                legendgroup=model_name,
            ),
            row=row,
            col=col,
        )

        # 5. Optional: Add median vertical line
        median_val = np.median(best_shot)
        fig.add_trace(
            go.Scatter(
                x=[median_val, median_val],
                y=[0, 1],
                mode="lines",
                line=dict(color=color, width=1, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
                legendgroup=model_name,
            ),
            row=row,
            col=col,
        )
