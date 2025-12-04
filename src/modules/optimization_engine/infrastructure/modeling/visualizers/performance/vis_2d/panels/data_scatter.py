import numpy as np
import plotly.graph_objects as go

def add_data_scatter(
    fig: go.Figure,
    row: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray | None,
    y_test: np.ndarray | None,
    input_symbol: str,
    output_symbol: str,
):
    """Adds scatter points for Train and Test data."""
    
    def _sym(symbol: str, idx: int) -> str:
        subscripts = {1: "\u2081", 2: "\u2082"}
        return f"{symbol}{subscripts.get(idx, idx)}"

    # Train Data
    for col_idx in [1, 2]:
        fig.add_trace(
            go.Scatter3d(
                x=X_train[:, 0],
                y=X_train[:, 1],
                z=y_train[:, col_idx - 1],
                mode="markers",
                name="Train Data",
                marker=dict(size=3, color="RoyalBlue", opacity=0.8),
                showlegend=(col_idx == 1),
                hovertemplate=f"<b>Train</b><br>{_sym(input_symbol, 1)}: %{{x:.4f}}<br>{_sym(input_symbol, 2)}: %{{y:.4f}}<br>{_sym(output_symbol, col_idx)}: %{{z:.4f}}<extra></extra>",
            ),
            row=row,
            col=col_idx,
        )

    # Test Data
    if X_test is not None and y_test is not None:
        for col_idx in [1, 2]:
            fig.add_trace(
                go.Scatter3d(
                    x=X_test[:, 0],
                    y=X_test[:, 1],
                    z=y_test[:, col_idx - 1],
                    mode="markers",
                    name="Test Data",
                    marker=dict(size=3, color="FireBrick", opacity=0.8, symbol="diamond"),
                    showlegend=(col_idx == 1),
                    hovertemplate=f"<b>Test</b><br>{_sym(input_symbol, 1)}: %{{x:.4f}}<br>{_sym(input_symbol, 2)}: %{{y:.4f}}<br>{_sym(output_symbol, col_idx)}: %{{z:.4f}}<extra></extra>",
                ),
                row=row,
                col=col_idx,
            )
