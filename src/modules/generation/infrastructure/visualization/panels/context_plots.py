import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..helpers.scatter_2d import (
    add_scatter_overlay,
    set_xy_limits,
)


def create_combined_context_figure(
    original_objectives: np.ndarray,
    target_objective: np.ndarray,
    candidate_objectives: np.ndarray,
    original_decisions: np.ndarray,
    candidate_decisions: np.ndarray,
    vertices_indices: list[int] | None = None,
) -> go.Figure:
    """
    Creates a single figure with two subplots:
    1. Decision Space (Generated vs Context x1, x2)
    2. Objective Space (Target vs Context)
    """
    # 0. Ensure inputs are numpy arrays (handles lists from serializable application results)
    original_objectives = np.asarray(original_objectives)
    target_objective = np.asarray(target_objective).flatten()
    candidate_objectives = np.asarray(candidate_objectives)
    original_decisions = np.asarray(original_decisions)
    candidate_decisions = np.asarray(candidate_decisions)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "<b>Decision Space (x1 vs x2)</b>",
            "<b>Objective Space (Raw)</b>",
        ),
        horizontal_spacing=0.15,
    )

    # --- SUBPLOT 1: DECISION SPACE ---
    # 1. Background Context Anchors
    add_scatter_overlay(
        fig,
        1,
        1,
        original_decisions[:, 0],
        original_decisions[:, 1],
        name="Context Anchors",
        symbol="circle",
        size=6,
        opacity=0.3,
        color="#bdc3c7",  # Silver
        show_legend=True,
    )

    # 1.5 Anchor Simplex (Decisions)
    if vertices_indices is not None and len(vertices_indices) > 0:
        poly_dec = original_decisions[vertices_indices]
        poly_dec = np.vstack([poly_dec, poly_dec[0]])  # Close the polygon
        fig.add_trace(
            go.Scatter(
                x=poly_dec[:, 0],
                y=poly_dec[:, 1],
                mode="lines+markers",
                name="Simplex Anchors (Dec)",
                line=dict(color="#e67e22", width=2, dash="dash"),  # Carrot (Orange)
                marker=dict(symbol="triangle-up", size=10, color="#e67e22"),
                fill="toself",
                fillcolor="rgba(230, 126, 34, 0.1)",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # 2. Generated Candidates (Decisions)
    add_scatter_overlay(
        fig,
        1,
        1,
        candidate_decisions[:, 0],
        candidate_decisions[:, 1],
        name="Generated (X)",
        symbol="circle",
        size=8,
        opacity=0.9,
        color="#d35400",  # Pumpkin (Orange)
    )

    # --- SUBPLOT 2: OBJECTIVE SPACE ---
    # 1. Background Context Objectives (Pareto Front)
    add_scatter_overlay(
        fig,
        1,
        2,
        original_objectives[:, 0],
        original_objectives[:, 1],
        name="Context Objectives",
        symbol="circle",
        size=6,
        opacity=0.3,
        color="#bdc3c7",  # Silver
    )

    # 1.5 Anchor Simplex (Objectives)
    if vertices_indices is not None and len(vertices_indices) > 0:
        poly_obj = original_objectives[vertices_indices]
        poly_obj = np.vstack([poly_obj, poly_obj[0]])  # Close the polygon
        fig.add_trace(
            go.Scatter(
                x=poly_obj[:, 0],
                y=poly_obj[:, 1],
                mode="lines+markers",
                name="Simplex Anchors (Obj)",
                line=dict(color="#e67e22", width=2, dash="dash"),  # Carrot (Orange)
                marker=dict(symbol="triangle-up", size=10, color="#e67e22"),
                fill="toself",
                fillcolor="rgba(230, 126, 34, 0.1)",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # 2. Generated Candidates (Objectives)
    add_scatter_overlay(
        fig,
        1,
        2,
        candidate_objectives[:, 0],
        candidate_objectives[:, 1],
        name="Generated (Obj)",
        symbol="circle",
        size=8,
        opacity=0.9,
        color="#2980b9",  # Belize Hole (Blue)
    )

    # 3. User Target
    target_obj_arr = np.array(target_objective).reshape(-1, 2)
    add_scatter_overlay(
        fig,
        1,
        2,
        target_obj_arr[:, 0],
        target_obj_arr[:, 1],
        name="User Target",
        symbol="star",
        size=15,
        opacity=1.0,
        color="#e74c3c",  # Alizarin (Red)
    )

    fig.update_layout(
        title=dict(
            text="<b>Generation Context Analysis</b>",
            font=dict(size=24),
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        height=600,
        width=1200,
        showlegend=True,
    )

    # Scaling Decision Plot (Col 1)
    all_dec_x = np.concatenate([original_decisions[:, 0], candidate_decisions[:, 0]])
    all_dec_y = np.concatenate([original_decisions[:, 1], candidate_decisions[:, 1]])
    set_xy_limits(fig, 1, 1, all_dec_x, all_dec_y)
    fig.update_xaxes(title_text="Decision Dim 1 (x1)", row=1, col=1)
    fig.update_yaxes(title_text="Decision Dim 2 (x2)", row=1, col=1)

    # Scaling Objective Plot (Col 2)
    all_obj_x = np.concatenate(
        [original_objectives[:, 0], target_obj_arr[:, 0], candidate_objectives[:, 0]]
    )
    all_obj_y = np.concatenate(
        [original_objectives[:, 1], target_obj_arr[:, 1], candidate_objectives[:, 1]]
    )
    set_xy_limits(fig, 1, 2, all_obj_x, all_obj_y)
    fig.update_xaxes(title_text="Objective 1", row=1, col=2)
    fig.update_yaxes(title_text="Objective 2", row=1, col=2)

    return fig
