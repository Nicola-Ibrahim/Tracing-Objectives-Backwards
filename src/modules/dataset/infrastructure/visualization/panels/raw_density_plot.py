import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..helpers.pdf_2d import add_pdf2d


def create_raw_decision_density_figure(historical_solutions: np.ndarray) -> go.Figure:
    """Creates 2D PDF density plot for raw decisions."""
    fig = make_subplots(rows=1, cols=1)

    add_pdf2d(
        fig,
        1,
        1,
        historical_solutions[:, 0],
        historical_solutions[:, 1],
        "$x_1$ (Raw)",
        "$x_2$ (Raw)",
        colorscale="Oranges",
        show_points=False,
    )

    fig.update_layout(
        title="<b>Raw Decision Density</b>",
        template="plotly_white",
        height=600,
        width=700,
    )
    return fig


def create_raw_objective_density_figure(historical_objectives: np.ndarray) -> go.Figure:
    """Creates 2D PDF density plot for raw objectives."""
    fig = make_subplots(rows=1, cols=1)

    add_pdf2d(
        fig,
        1,
        1,
        historical_objectives[:, 0],
        historical_objectives[:, 1],
        "$y_1$ (Raw)",
        "$y_2$ (Raw)",
        colorscale="Blues",
        show_points=False,
    )

    fig.update_layout(
        title="<b>Raw Objective Density</b>",
        template="plotly_white",
        height=600,
        width=700,
    )
    return fig
