from typing import Any

import plotly.graph_objects as go


def add_metrics_bar_plots(
    fig: go.Figure,
    results_map: dict[str, dict[str, Any]],
    color_map: dict[str, str],
    model_names: list[str],
) -> None:
    """
    Adds metrics bar charts (Best Shot, Calibration Error, Diversity) to Row 2.
    Each model is added as a separate trace and linked to the legend via legendgroup.
    """

    # 1. Best Shot (Row 2, Col 1)
    for model_name in model_names:
        if model_name in results_map:
            metrics = results_map[model_name]["metrics"]
            color = color_map.get(model_name, "gray")

            fig.add_trace(
                go.Bar(
                    x=[model_name],
                    y=[metrics["best_shot_error"]],
                    marker_color=color,
                    showlegend=False,
                    legendgroup=model_name,
                    name="Best Shot Error",
                ),
                row=2,
                col=1,
            )

    # 2. Calibration Error (Row 2, Col 2)
    for model_name in model_names:
        if model_name in results_map:
            metrics = results_map[model_name]["metrics"]
            color = color_map.get(model_name, "gray")

            fig.add_trace(
                go.Bar(
                    x=[model_name],
                    y=[metrics.get("calibration_error", 0)],
                    marker_color=color,
                    showlegend=False,
                    legendgroup=model_name,
                    name="Calibration Error",
                ),
                row=2,
                col=2,
            )

    # 3. CRPS (Row 2, Col 3)
    for model_name in model_names:
        if model_name in results_map:
            metrics = results_map[model_name]["metrics"]
            color = color_map.get(model_name, "gray")

            fig.add_trace(
                go.Bar(
                    x=[model_name],
                    y=[metrics.get("crps", 0)],
                    marker_color=color,
                    showlegend=False,
                    legendgroup=model_name,
                    name="CRPS",
                ),
                row=2,
                col=3,
            )

    # 4. Diversity (Row 3, Col 1)
    for model_name in model_names:
        if model_name in results_map:
            metrics = results_map[model_name]["metrics"]
            color = color_map.get(model_name, "gray")

            fig.add_trace(
                go.Bar(
                    x=[model_name],
                    y=[metrics["diversity_score"]],
                    marker_color=color,
                    showlegend=False,
                    legendgroup=model_name,
                    name="Diversity Score",
                ),
                row=3,
                col=1,
            )

    # 5. Sharpness (Row 3, Col 2)
    for model_name in model_names:
        if model_name in results_map:
            metrics = results_map[model_name]["metrics"]
            color = color_map.get(model_name, "gray")

            fig.add_trace(
                go.Bar(
                    x=[model_name],
                    y=[metrics.get("sharpness", 0)],
                    marker_color=color,
                    showlegend=False,
                    legendgroup=model_name,
                    name="Sharpness",
                ),
                row=3,
                col=2,
            )
