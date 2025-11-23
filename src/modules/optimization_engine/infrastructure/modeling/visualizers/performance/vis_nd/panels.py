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
    X_train_red,
    y_train_1d,
    X_test_red=None,
    y_test_1d=None,
    col: int = 1,
) -> None:
    """
    Overlay training (and optional test) points on the 1D plot with standard colors.
    """
    # Train points
    fig.add_trace(
        go.Scatter(
            x=X_train_red.ravel(),
            y=y_train_1d.ravel(),
            mode="markers",
            name="Train Data",
            marker=dict(size=4, color="RoyalBlue", opacity=0.6),
            hovertemplate="<b>Train</b><br>x_red: %{x:.4f}<br>y: %{y:.4f}<extra></extra>",
        ),
        row=row,
        col=col,
    )

    # Test points
    if X_test_red is not None and y_test_1d is not None:
        fig.add_trace(
            go.Scatter(
                x=X_test_red.ravel(),
                y=y_test_1d.ravel(),
                mode="markers",
                name="Test Data",
                marker=dict(size=4, color="FireBrick", opacity=0.6, symbol="x"),
                hovertemplate="<b>Test</b><br>x_red: %{x:.4f}<br>y: %{y:.4f}<extra></extra>",
            ),
            row=row,
            col=col,
        )
