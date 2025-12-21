from typing import Any

import plotly.graph_objects as go


def add_metrics_bar_plots(
    fig: go.Figure,
    results_map: dict[str, dict[str, Any]],
    color_map: dict[str, str],
    model_names: list[str],
) -> None:
    """
    Adds metrics bar charts (Best Shot, Reliability, Diversity) to Row 2.
    """
    best_shots_vals = []
    calibration_error_vals = []
    diversity_vals = []

    for model_name in model_names:
        metrics = results_map[model_name]["metrics"]
        best_shots_vals.append(metrics["best_shot_error"])
        calibration_error_vals.append(metrics["calibration_error"])
        diversity_vals.append(metrics["diversity_score"])

    # Row 2, Col 1: Best Shot
    fig.add_trace(
        go.Bar(
            x=model_names,
            y=best_shots_vals,
            marker_color=[color_map.get(m, "gray") for m in model_names],
            showlegend=False,
            name="Best Shot Error",
        ),
        row=2,
        col=1,
    )

    # Row 2, Col 2: Calibration Error
    fig.add_trace(
        go.Bar(
            x=model_names,
            y=calibration_error_vals,
            marker_color=[color_map.get(m, "gray") for m in model_names],
            showlegend=False,
            name="Calibration Error",
        ),
        row=2,
        col=2,
    )

    # Row 2, Col 3: Diversity
    fig.add_trace(
        go.Bar(
            x=model_names,
            y=diversity_vals,
            marker_color=[color_map.get(m, "gray") for m in model_names],
            showlegend=False,
            name="Diversity Score",
        ),
        row=2,
        col=3,
    )
