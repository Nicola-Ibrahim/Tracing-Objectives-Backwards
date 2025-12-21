import numpy as np
import plotly.graph_objects as go

from .grid import build_grid
from .model_vis import add_model_visualization
from .data_scatter import add_data_scatter
from .scene_config import configure_3d_scenes

def add_surfaces_2d(
    fig: go.Figure,
    *,
    row: int,
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray | None,
    y_test: np.ndarray | None,
    grid_res: int = 50,
    input_symbol: str = "x",
    output_symbol: str = "y",
) -> None:
    
    # 1. Build Grid
    GX1, GX2, X_grid = build_grid(X_train, grid_res)

    # 2. Plot Model (Ghost Scatter or Surface)
    # We capture the Z-values used for plotting to help set axis ranges later
    z_plot_vals = add_model_visualization(
        fig, row, estimator, X_grid, GX1, GX2, input_symbol, output_symbol, grid_res
    )

    # 3. Plot Data Points (Train/Test)
    add_data_scatter(
        fig, row, X_train, y_train, X_test, y_test, input_symbol, output_symbol
    )

    # 4. Configure Axes
    configure_3d_scenes(
        fig, row, X_train, X_test, y_train, y_test, GX1, GX2, z_plot_vals, input_symbol, output_symbol
    )
