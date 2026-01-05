from typing import Any

import plotly.graph_objects as go


def add_calibration_plot(
    fig: go.Figure,
    row: int,
    col: int,
    results_map: dict[str, dict[str, Any]],
    color_map: dict[str, str],
) -> None:
    """
    Adds Calibration Curve (PIT) to the specified subplot using pre-calculated data.
    """
    # Add ideal line
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

    for model_name, res in results_map.items():
        calibration = res.get("calibration")
        if calibration is None:
            continue

        color = color_map.get(model_name, "gray")
        pit_values = calibration["pit_values"]
        cdf_y = calibration["cdf_y"]

        fig.add_trace(
            go.Scatter(
                x=pit_values,
                y=cdf_y,
                mode="lines",
                name=f"{model_name}",
                line=dict(color=color, width=2),
                opacity=0.7,
                legendgroup=model_name,
            ),
            row=row,
            col=col,
        )

    # Annotations
    fig.add_annotation(
        x=0.8,
        y=0.2,
        text="Overconfident",
        showarrow=False,
        font=dict(size=10, color="gray"),
        row=row,
        col=col,
    )
    fig.add_annotation(
        x=0.2,
        y=0.8,
        text="Underconfident",
        showarrow=False,
        font=dict(size=10, color="gray"),
        row=row,
        col=col,
    )
