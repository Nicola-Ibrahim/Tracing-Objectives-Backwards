import numpy as np
import plotly.graph_objects as go

def configure_3d_scenes(
    fig: go.Figure,
    row: int,
    X_train: np.ndarray,
    X_test: np.ndarray | None,
    y_train: np.ndarray,
    y_test: np.ndarray | None,
    GX1: np.ndarray,
    GX2: np.ndarray,
    z_plot_vals: tuple[np.ndarray, np.ndarray],
    input_symbol: str,
    output_symbol: str,
):
    """Configures axis ranges, titles, and aspect ratios for the 3D scenes."""
    
    def _sym(symbol: str, idx: int) -> str:
        subscripts = {1: "\u2081", 2: "\u2082"}
        return f"{symbol}{subscripts.get(idx, idx)}"

    def _minmax(*arrs):
        vals = [np.asarray(a).ravel() for a in arrs if a is not None]
        if not vals:
            return (0.0, 1.0)
        v = np.concatenate(vals)
        v = v[np.isfinite(v)]
        return (float(v.min()), float(v.max())) if v.size else (0.0, 1.0)

    def _pad(a, b, frac=0.02):
        d = (b - a) if b > a else 1.0
        return (a - frac * d, b + frac * d)

    def _ticks(rng, n=5):
        return list(np.round(np.linspace(rng[0], rng[1], n), 5))

    z_y1, z_y2 = z_plot_vals

    # common XY ranges (include grid + any test Xs)
    x_rng = _minmax(X_train[:, 0], (X_test[:, 0] if X_test is not None else None), GX1)
    y_rng = _minmax(X_train[:, 1], (X_test[:, 1] if X_test is not None else None), GX2)

    # Z ranges from Y + predicted surfaces/clouds
    z1_min, z1_max = _minmax(
        y_train[:, 0], (y_test[:, 0] if y_test is not None else None), z_y1
    )
    z2_min, z2_max = _minmax(
        y_train[:, 1], (y_test[:, 1] if y_test is not None else None), z_y2
    )
    z_rng = (min(z1_min, z2_min), max(z1_max, z2_max))

    x_rng = _pad(*x_rng)
    y_rng = _pad(*y_rng)
    z_rng = _pad(*z_rng)

    scene_common = dict(
        xaxis=dict(
            title=_sym(input_symbol, 1),
            range=list(x_rng),
            tickmode="array",
            tickvals=_ticks(x_rng),
            backgroundcolor="rgb(245, 245, 245)",
            gridcolor="white",
            showbackground=True,
            zerolinecolor="white",
        ),
        yaxis=dict(
            title=_sym(input_symbol, 2),
            range=list(y_rng),
            tickmode="array",
            tickvals=_ticks(y_rng),
            backgroundcolor="rgb(245, 245, 245)",
            gridcolor="white",
            showbackground=True,
            zerolinecolor="white",
        ),
        aspectmode="cube",
    )

    scene1_axes = scene_common.copy()
    scene1_axes.update(
        zaxis=dict(
            title=_sym(output_symbol, 1),
            range=list(z_rng),
            tickmode="array",
            tickvals=_ticks(z_rng),
            backgroundcolor="rgb(245, 245, 245)",
            gridcolor="white",
            showbackground=True,
            zerolinecolor="white",
        )
    )

    scene2_axes = scene_common.copy()
    scene2_axes.update(
        zaxis=dict(
            title=_sym(output_symbol, 2),
            range=list(z_rng),
            tickmode="array",
            tickvals=_ticks(z_rng),
            backgroundcolor="rgb(245, 245, 245)",
            gridcolor="white",
            showbackground=True,
            zerolinecolor="white",
        )
    )

    # Apply identical axes to both scenes
    fig.update_scenes(row=row, col=1, **scene1_axes)
    fig.update_scenes(row=row, col=2, **scene2_axes)
