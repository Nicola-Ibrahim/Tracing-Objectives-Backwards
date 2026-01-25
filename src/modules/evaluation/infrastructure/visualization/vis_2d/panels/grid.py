import numpy as np

def build_grid(X_train: np.ndarray, grid_res: int):
    """Create a meshgrid over the training domain."""
    x1_min, x1_max = float(np.min(X_train[:, 0])), float(np.max(X_train[:, 0]))
    x2_min, x2_max = float(np.min(X_train[:, 1])), float(np.max(X_train[:, 1]))
    gx1 = np.linspace(x1_min, x1_max, grid_res)
    gx2 = np.linspace(x2_min, x2_max, grid_res)
    GX1, GX2 = np.meshgrid(gx1, gx2, indexing="xy")
    X_grid = np.stack([GX1.ravel(), GX2.ravel()], axis=1)
    return GX1, GX2, X_grid
