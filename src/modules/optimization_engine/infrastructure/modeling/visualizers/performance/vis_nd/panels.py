import numpy as np
import plotly.graph_objects as go


def add_fit_1d(
    fig: go.Figure,
    *,
    row: int,
    X_red: np.ndarray,
    center: np.ndarray,
    p05: np.ndarray,
    p95: np.ndarray,
    name_center: str = "Center",
) -> None:
    order = np.argsort(X_red)
    fig.add_trace(
        go.Scatter(
            x=X_red[order], 
            y=center[order], 
            mode="lines", 
            name=name_center,
            line=dict(color="RoyalBlue", width=3),
            hovertemplate="<b>%{y:.4f}</b><extra></extra>"
        ),
        row=row,
        col=1,
    )
    ribbon_x = np.concatenate([X_red[order], X_red[order][::-1]])
    ribbon_y = np.concatenate([p95[order], p05[order][::-1]])
    fig.add_trace(
        go.Scatter(
            x=ribbon_x,
            y=ribbon_y,
            fill="toself",
            fillcolor="rgba(65, 105, 225, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="5–95% Conf",
            hoverinfo="skip",
        ),
        row=row,
        col=1,
    )


def add_points_overlay(
    fig: go.Figure,
    *,
    row: int,
    X_train_red: np.ndarray | None,
    y_train_1d: np.ndarray | None,
    X_test_red: np.ndarray | None,
    y_test_1d: np.ndarray | None,
) -> None:
    if X_train_red is not None and y_train_1d is not None:
        fig.add_trace(
            go.Scatter(
                x=X_train_red,
                y=y_train_1d[:, 0],
                mode="markers",
                name="Train",
                marker=dict(opacity=0.5, size=5, color="black"),
                hovertemplate="<b>Train</b>: %{y:.4f}<extra></extra>",
            ),
            row=row,
            col=1,
        )
    if X_test_red is not None and y_test_1d is not None:
        fig.add_trace(
            go.Scatter(
                x=X_test_red,
                y=y_test_1d[:, 0],
                mode="markers",
                name="Test",
                marker=dict(opacity=0.7, size=6, color="red", symbol="diamond"),
                hovertemplate="<b>Test</b>: %{y:.4f}<extra></extra>",
            ),
            row=row,
            col=1,
        )
