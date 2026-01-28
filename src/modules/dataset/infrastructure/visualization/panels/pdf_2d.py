import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde


def add_pdf2d(
    fig: go.Figure,
    row: int,
    col: int,
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str,
    colorscale: str = "Blues",
    reverse_scale: bool = True,
    line_color: str = "#0b0b0b",
    point_size: int = 4,
    point_edge_color: str = "rgba(255,255,255,0.7)",
    show_points: bool = False,
):
    # keep only finite & align
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    n = min(len(x), len(y))
    if n == 0:
        return
    x, y = x[:n], y[:n]

    # helper: build grid + KDE
    def _kde_on(xv, yv, q_lo=0.01, q_hi=0.99, ngrid=160):
        kde = gaussian_kde(np.vstack([xv, yv]))
        # compact grid: robust bounds by quantiles
        x_lo, x_hi = float(np.quantile(xv, q_lo)), float(np.quantile(xv, q_hi))
        y_lo, y_hi = float(np.quantile(yv, q_lo)), float(np.quantile(yv, q_hi))
        # small padding
        pad_x = 0.03 * max(1e-9, x_hi - x_lo)
        pad_y = 0.03 * max(1e-9, y_hi - y_lo)
        gx = np.linspace(x_lo - pad_x, x_hi + pad_x, ngrid)
        gy = np.linspace(y_lo - pad_y, y_hi + pad_y, ngrid)
        XX, YY = np.meshgrid(gx, gy)
        ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
        # density at points (used to color points)
        z_pts = kde(np.vstack([x, y]))
        return gx, gy, ZZ, z_pts

    # try KDE, retry with tiny jitter if needed; otherwise fallback to hist2d
    try:
        gx, gy, ZZ, z_pts = _kde_on(x, y)
    except Exception:
        try:
            epsx = 1e-6 * (np.std(x) + 1e-12)
            epsy = 1e-6 * (np.std(y) + 1e-12)
            gx, gy, ZZ, z_pts = _kde_on(
                x + np.random.normal(0, epsx, size=x.shape),
                y + np.random.normal(0, epsy, size=y.shape),
            )
        except Exception:
            fig.add_trace(
                go.Histogram2d(
                    x=x,
                    y=y,
                    nbinsx=40,
                    nbinsy=40,
                    colorscale=colorscale,
                    reversescale=reverse_scale,
                    histnorm="probability density",
                    showscale=False,
                    name="2D Histogram",
                ),
                row=row,
                col=col,
            )
            fig.update_xaxes(title_text=x_label, row=row, col=col)
            fig.update_yaxes(title_text=y_label, row=row, col=col)
            return

    # robust z-limits for nice contrast
    flat = ZZ.ravel()
    zmin = float(np.quantile(flat, 0.02))
    zmax = float(np.quantile(flat, 0.98))
    if zmax <= zmin:
        zmin, zmax = float(flat.min()), float(flat.max())
    levels = 9
    step = (zmax - zmin) / max(levels, 1)

    # Filled, labeled contours (blue) — like your reference
    fig.add_trace(
        go.Contour(
            x=gx,
            y=gy,
            z=ZZ,
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale,
            reversescale=reverse_scale,
            contours=dict(
                coloring="fill",
                showlines=True,
                showlabels=True,
                start=zmin,
                end=zmax,
                size=step,
                labelfont=dict(size=10, color="#111"),
            ),
            line=dict(color=line_color, width=1.1),
            colorbar=dict(title="KDE", tickformat=".3f"),
            name="KDE",
            hovertemplate="p(x,y)=%{z:.4f}<extra></extra>",
            showscale=False,
        ),
        row=row,
        col=col,
    )

    # Points colored by their KDE value (adds a “glow” without clutter)
    if show_points:
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=z_pts,
                    colorscale=colorscale,
                    reversescale=reverse_scale,
                    opacity=0.9,
                    line=dict(color=point_edge_color, width=0.5),
                ),
                name="Points",
                showlegend=False,
                hovertemplate=(
                    f"{x_label}: %{{x:.4f}}<br>"
                    f"{y_label}: %{{y:.4f}}<br>"
                    "p: %{{marker.color:.4f}}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    # Axes titles + equal aspect so the shape isn’t distorted
    fig.update_xaxes(title_text=x_label, row=row, col=col)
    fig.update_yaxes(title_text=y_label, row=row, col=col)
    subplot_idx = (row - 1) * 2 + col
    fig.update_yaxes(scaleanchor=f"x{subplot_idx}", row=row, col=col)
