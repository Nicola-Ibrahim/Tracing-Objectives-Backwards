from typing import Any, List

import numpy as np
import plotly.graph_objects as go


def add_accuracy_bias_dispersion_plot(
    fig: go.Figure,
    results_map: dict[str, dict[str, Any]],
    model_names: List[str],
) -> None:
    """
    Adds a grouped box plot (approximating a Boxen plot) for Bias and Dispersion
    metrics across all estimators.
    """
    # 1. Colors
    COLOR_BIAS = "#003f5c"  # Deep Navy Blue
    COLOR_DISPERSION = "#488a5c"  # Muted Sage Green

    # 2. Add traces for each estimator
    for model_name in model_names:
        if model_name not in results_map:
            continue

        acc = results_map[model_name].get("accuracy", {})
        b = np.array(acc.get("systematic_bias", []))
        v = np.array(acc.get("cloud_dispersion", []))

        if len(b) == 0 or len(v) == 0:
            continue

        # Add Bias Trace
        fig.add_trace(
            go.Box(
                y=b,
                x=[model_name] * len(b),
                name="Bias (Systematic Error)",
                marker=dict(color=COLOR_BIAS, opacity=0.3, size=3),
                line=dict(width=1.5),
                boxpoints="outliers",
                offsetgroup="Bias",
                legendgroup="Bias",
                showlegend=(model_name == model_names[0]),
            )
        )

        # Add Dispersion Trace
        fig.add_trace(
            go.Box(
                y=v,
                x=[model_name] * len(v),
                name="Dispersion (Random Error)",
                marker=dict(color=COLOR_DISPERSION, opacity=0.3, size=3),
                line=dict(width=1.5),
                boxpoints="outliers",
                offsetgroup="Dispersion",
                legendgroup="Dispersion",
                showlegend=(model_name == model_names[0]),
            )
        )

    # 3. Configure Grouping and Global Aesthetics
    fig.update_layout(
        boxmode="group",
        template="plotly_white",
        yaxis=dict(
            title="Error Magnitude (L2 Norm)",
            gridcolor="lightgrey",
            zerolinecolor="lightgrey",
            range=[0, 0.2],  # As requested: 0.0 to 0.2
        ),
        xaxis=dict(
            title="Estimator Version",
        ),
        legend=dict(
            title="Error Component",
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
        ),
    )

    # Remove top and right spines (approximated in plotly)
    fig.update_xaxes(showline=True, linewidth=1, linecolor="lightgrey", mirror=False)
    fig.update_yaxes(showline=True, linewidth=1, linecolor="lightgrey", mirror=False)
