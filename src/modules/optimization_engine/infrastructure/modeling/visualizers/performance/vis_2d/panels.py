import numpy as np
import plotly.graph_objects as go

from ..common.prediction import point_predict


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
) -> None:
    # --- build grid over TRAIN objective domain (x1,x2) ---
    x1_min, x1_max = float(np.min(X_train[:, 0])), float(np.max(X_train[:, 0]))
    x2_min, x2_max = float(np.min(X_train[:, 1])), float(np.max(X_train[:, 1]))
    gx1 = np.linspace(x1_min, x1_max, grid_res)
    gx2 = np.linspace(x2_min, x2_max, grid_res)
    GX1, GX2 = np.meshgrid(gx1, gx2, indexing="xy")
    X_grid = np.stack([GX1.ravel(), GX2.ravel()], axis=1)

    # --- use point_predict for consistent semantics (MAP -> mean -> predict) ---
    Yg = point_predict(estimator, X_grid)  # returns (n, 2)
    z_y1 = Yg[:, 0].reshape(grid_res, grid_res)
    z_y2 = Yg[:, 1].reshape(grid_res, grid_res)

    # -------- row1: surfaces + TRAIN/TEST point clouds (x on axes; y as height) -----
    fig.add_trace(
        go.Surface(
            x=GX1, y=GX2, z=z_y1, opacity=0.45, showscale=False, name="Decision y1(x)"
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Surface(
            x=GX1, y=GX2, z=z_y2, opacity=0.45, showscale=False, name="Decision y2(x)"
        ),
        row=row,
        col=2,
    )

    # TRAIN points
    fig.add_trace(
        go.Scatter3d(
            x=X_train[:, 0],
            y=X_train[:, 1],
            z=y_train[:, 0],
            mode="markers",
            name="Train decisions (y1)",
            marker=dict(size=3, opacity=0.5),
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=X_train[:, 0],
            y=X_train[:, 1],
            z=y_train[:, 1],
            mode="markers",
            name="Train decisions (y2)",
            marker=dict(size=3, opacity=0.5),
        ),
        row=row,
        col=2,
    )

    # TEST points (if provided)
    if X_test is not None and y_test is not None:
        fig.add_trace(
            go.Scatter3d(
                x=X_test[:, 0],
                y=X_test[:, 1],
                z=y_test[:, 0],
                mode="markers",
                name="Test decisions (y1)",
                marker=dict(size=3, opacity=0.5),
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=X_test[:, 0],
                y=X_test[:, 1],
                z=y_test[:, 1],
                mode="markers",
                name="Test decisions (y2)",
                marker=dict(size=3, opacity=0.5),
            ),
            row=row,
            col=2,
        )

    # --- unified axis ranges & consistent aspect/ticks for both 3D scenes ---
    def _minmax(*arrs):
        vals = [np.asarray(a).ravel() for a in arrs if a is not None]
        if not vals:
            return (0.0, 1.0)
        v = np.concatenate(vals)
        v = v[np.isfinite(v)]
        return (float(v.min()), float(v.max())) if v.size else (0.0, 1.0)

    # common XY ranges (include grid + any test Xs)
    x_rng = _minmax(X_train[:, 0], (X_test[:, 0] if X_test is not None else None), GX1)
    y_rng = _minmax(X_train[:, 1], (X_test[:, 1] if X_test is not None else None), GX2)

    # Z ranges from Y + predicted surfaces
    z1_min, z1_max = _minmax(
        y_train[:, 0], (y_test[:, 0] if y_test is not None else None), z_y1
    )
    z2_min, z2_max = _minmax(
        y_train[:, 1], (y_test[:, 1] if y_test is not None else None), z_y2
    )
    z_rng = (min(z1_min, z2_min), max(z1_max, z2_max))

    # small padding so points aren’t glued to box
    def _pad(a, b, frac=0.02):
        d = (b - a) if b > a else 1.0
        return (a - frac * d, b + frac * d)

    x_rng = _pad(*x_rng)
    y_rng = _pad(*y_rng)
    z_rng = _pad(*z_rng)

    # consistent ticks
    def _ticks(rng, n=5):
        return list(np.round(np.linspace(rng[0], rng[1], n), 5))

    scene1_axes = dict(
        xaxis=dict(
            title="Objective x1 (norm)",
            range=list(x_rng),
            tickmode="array",
            tickvals=_ticks(x_rng),
        ),
        yaxis=dict(
            title="Objective x2 (norm)",
            range=list(y_rng),
            tickmode="array",
            tickvals=_ticks(y_rng),
        ),
        zaxis=dict(
            title="Decision y1 (norm)",
            range=list(z_rng),
            tickmode="array",
            tickvals=_ticks(z_rng),
        ),
        aspectmode="cube",  # equal scaling on x/y/z
    )
    scene2_axes = dict(
        xaxis=dict(
            title="Objective x1 (norm)",
            range=list(x_rng),
            tickmode="array",
            tickvals=_ticks(x_rng),
        ),
        yaxis=dict(
            title="Objective x2 (norm)",
            range=list(y_rng),
            tickmode="array",
            tickvals=_ticks(y_rng),
        ),
        zaxis=dict(
            title="Decision y2 (norm)",
            range=list(z_rng),
            tickmode="array",
            tickvals=_ticks(z_rng),
        ),
        aspectmode="cube",
    )

    # Apply identical axes to both scenes (recommended approach: update_scenes)
    fig.update_scenes(
        row=row, col=1, **scene1_axes
    )  # control 3D scene ranges/ticks/aspect here, not per-trace. :contentReference[oaicite:2]{index=2}
    fig.update_scenes(row=row, col=2, **scene2_axes)
