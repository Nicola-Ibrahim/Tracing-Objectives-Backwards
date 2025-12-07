import numpy as np
import plotly.graph_objects as go


def add_model_visualization(
    fig: go.Figure,
    row: int,
    estimator,
    X_grid: np.ndarray,
    GX1: np.ndarray,
    GX2: np.ndarray,
    input_symbol: str,
    output_symbol: str,
    grid_res: int,
):
    """Plots Ghost 3D Scatter for probabilistic models.

    Visualizes uncertainty in the model predictions by sampling multiple
    times from the probabilistic distribution and plotting as a ghost scatter.
    """

    # Sample from probabilistic estimator
    n_samples = 50
    Y_samples = estimator.sample(X_grid, n_samples=n_samples)

    out_dim = Y_samples.shape[-1]

    # Flatten for plotting
    Y_flat = Y_samples.reshape(-1, out_dim)
    X_flat = np.repeat(X_grid, n_samples, axis=0)

    x_plot = X_flat[:, 0]
    y_plot = X_flat[:, 1]
    z_y1 = Y_flat[:, 0]
    z_y2 = Y_flat[:, 1] if out_dim > 1 else np.zeros_like(z_y1)

    # Plot Ghost Clouds
    _add_ghost_trace(fig, row, 1, x_plot, y_plot, z_y1, input_symbol, output_symbol, 1)
    if out_dim > 1:
        _add_ghost_trace(
            fig, row, 2, x_plot, y_plot, z_y2, input_symbol, output_symbol, 2
        )

    return (z_y1, z_y2)


def _add_ghost_trace(fig, row, col, x, y, z, input_symbol, output_symbol, out_idx):
    def _sym(symbol: str, idx: int) -> str:
        subscripts = {1: "\u2081", 2: "\u2082"}
        return f"{symbol}{subscripts.get(idx, idx)}"

    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            name=f"{_sym(output_symbol, out_idx)} Uncertainty",
            marker=dict(size=2, color=z, colorscale="Cividis", opacity=0.1),
            hovertemplate=f"<b>{_sym(input_symbol, 1)}</b>: %{{x:.4f}}<br><b>{_sym(input_symbol, 2)}</b>: %{{y:.4f}}<br><b>{_sym(output_symbol, out_idx)}</b>: %{{z:.4f}}<extra></extra>",
        ),
        row=row,
        col=col,
    )
