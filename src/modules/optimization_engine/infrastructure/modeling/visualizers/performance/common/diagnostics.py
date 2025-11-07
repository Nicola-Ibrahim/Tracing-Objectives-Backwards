import numpy as np
import plotly.graph_objects as go


def _finite(values) -> np.ndarray:
    """Return a 1-D float array containing only finite entries."""
    arr = np.asarray(values).astype(float).ravel()
    return arr[np.isfinite(arr)]


def _percentile_limits(
    values, lower: float = 1.0, upper: float = 99.0
) -> tuple[float, float]:
    """Compute robust bounds using lower/upper percentiles."""
    arr = _finite(values)
    if arr.size == 0:
        return (-1.0, 1.0)
    lo, hi = float(np.percentile(arr, lower)), float(np.percentile(arr, upper))
    if lo == hi:
        delta = abs(lo) if lo else 1.0
        return (lo - 0.1 * delta, hi + 0.1 * delta)
    return (lo, hi)


def _symmetric_limits(values, percentile: float = 99.0) -> tuple[float, float]:
    """Return symmetric +/- bounds using a percentile of |values|."""
    arr = _finite(values)
    if arr.size == 0:
        return (-1.0, 1.0)
    bound = float(np.percentile(np.abs(arr), percentile))
    bound = bound if bound else 1.0
    return (-bound, bound)


def _pad_range(bounds: tuple[float, float], frac: float = 0.05) -> tuple[float, float]:
    """Pad bounds by a fractional margin to avoid clipping markers."""
    lo, hi = bounds
    span = hi - lo if hi > lo else 1.0
    return (lo - frac * span, hi + frac * span)


def add_residuals_vs_fitted(
    fig: go.Figure,
    *,
    row: int,
    fitted,
    resid,
    label: str,
    col: int = 1,
) -> None:
    """
    Plot residuals vs fitted values.

    Args:
        fig: Plotly figure to mutate.
        row: Target subplot row (1-indexed).
        fitted: Predicted values aligned with residuals.
        resid: Residuals (observed - predicted).
        label: Legend suffix identifying the split (e.g. "train").
        col: Target subplot column (default: 1).
    """
    # Derive consistent axis limits so comparisons remain fair across estimators.
    fitted_limits = _pad_range(_percentile_limits(fitted))
    resid_limits = _pad_range(_symmetric_limits(resid))

    # Scatter each residual vs its fitted value.
    fig.add_trace(
        go.Scatter(
            x=fitted,
            y=resid,
            mode="markers",
            name=f"Residuals ({label})",
            marker=dict(opacity=0.35, size=6),
        ),
        row=row,
        col=col,
    )
    # Draw a zero-residual baseline.
    fig.add_trace(
        go.Scatter(
            x=list(fitted_limits),
            y=[0.0, 0.0],
            mode="lines",
            name="Zero residual",
            line=dict(color="black", width=1),
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    # Apply the normalized ranges to both axes.
    fig.update_xaxes(
        title_text="Fitted (norm)", range=list(fitted_limits), row=row, col=col
    )
    fig.update_yaxes(
        title_text="Residual (norm)", range=list(resid_limits), row=row, col=col
    )


def add_residual_hist(
    fig: go.Figure,
    *,
    row: int,
    resid,
    label: str,
    col: int = 1,
) -> None:
    """
    Plot a residual histogram with density scaling.

    Args:
        fig: Plotly figure to mutate.
        row: Target subplot row (1-indexed).
        resid: Residual values to bin.
        label: Legend suffix identifying the split.
        col: Target subplot column (default: 1).
    """
    # Use symmetric limits matching the scatter view.
    resid_limits = _pad_range(_symmetric_limits(resid))
    # Histogram plotted as density for apples-to-apples comparison across splits.
    fig.add_trace(
        go.Histogram(
            x=resid,
            nbinsx=40,
            histnorm="probability density",
            name=f"Resid ({label})",
            opacity=0.7,
        ),
        row=row,
        col=col,
    )
    # Highlight zero to make bias easy to spot.
    vals = resid[np.isfinite(resid)]
    if vals.size:
        counts, _ = np.histogram(vals, bins=40, density=True)
        ymax = float(np.nanmax(counts)) if counts.size else 1.0
        fig.add_trace(
            go.Scatter(
                x=[0.0, 0.0],
                y=[0.0, ymax * 1.05],
                mode="lines",
                name="Zero",
                line=dict(color="black", width=1),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    # Clamp x-axis to the shared residual limits.
    fig.update_xaxes(
        title_text="Residual (norm)", range=list(resid_limits), row=row, col=col
    )
    fig.update_yaxes(title_text="Density", row=row, col=col)


def add_joint_residual(
    fig: go.Figure,
    *,
    row: int,
    col: int,
    resid_y1,
    resid_y2,
) -> None:
    """
    Plot a joint density of two residual dimensions.

    Args:
        fig: Plotly figure to mutate.
        row: Target subplot row.
        col: Target subplot column.
        resid_y1: Residuals for the first dimension.
        resid_y2: Residuals for the second dimension.
    """
    # Filter NaNs so contours render cleanly.
    vx = resid_y1[np.isfinite(resid_y1)]
    vy = resid_y2[np.isfinite(resid_y2)]
    # Heatmap contours show joint density.
    fig.add_trace(
        go.Histogram2dContour(
            x=vx,
            y=vy,
            ncontours=20,
            contours_coloring="heatmap",
            showscale=False,
            name="Residual density",
            opacity=0.85,
        ),
        row=row,
        col=col,
    )
    # Scatter overlay reveals the actual residual samples.
    fig.add_trace(
        go.Scatter(
            x=vx,
            y=vy,
            mode="markers",
            name="Residuals",
            marker=dict(size=4, opacity=0.25),
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    fig.update_xaxes(title_text="Residual y1 (norm)", row=row, col=col)
    fig.update_yaxes(title_text="Residual y2 (norm)", row=row, col=col)


def add_loss_curves(
    fig: go.Figure,
    *,
    row: int,
    loss_history: dict | None,
    col: int = 1,
) -> None:
    """
    Draw training/validation/test loss curves.

    Args:
        fig: Plotly figure to mutate.
        row: Target subplot row.
        loss_history: Loss history payload (dict or model) compatible with .dict().
        col: Target subplot column (default: 1).
    """
    if loss_history is None:
        return
    # Accept dict-like objects (Pydantic, dataclass, plain dict).
    lh = loss_history
    if hasattr(lh, "model_dump"):
        lh = lh.model_dump()
    elif hasattr(lh, "dict"):
        lh = lh.dict()
    if not isinstance(lh, dict):
        return
    # Extract time/bin axes and loss arrays (empty list fallback keeps logic simple).
    bins = list(lh.get("bins", []))
    train = list(lh.get("train_loss", []))
    val = list(lh.get("val_loss", []))
    test = list(lh.get("test_loss", []))
    n_tr = list(lh.get("n_train", []))
    bin_type: str = str(lh.get("bin_type", "bin"))
    # Select an x-axis label/value sequence that matches the binning strategy.
    if bin_type == "train_fraction" and n_tr:
        x_vals, x_label = n_tr, "Number of Training Samples"
    elif bin_type == "epoch":
        x_vals, x_label = (bins if bins else list(range(1, len(train) + 1))), "Epoch"
    else:
        x_vals = bins if bins else list(range(len(train)))
        x_label = {"param": "Parameter", "bin": "Bin"}.get(bin_type, bin_type.title())
    # Plot each curve only if values exist.
    if train:
        fig.add_trace(
            go.Scatter(x=x_vals, y=train, mode="lines+markers", name="Train"),
            row=row,
            col=col,
        )
    if any(v is not None for v in val):
        fig.add_trace(
            go.Scatter(x=x_vals, y=val, mode="lines+markers", name="Validation"),
            row=row,
            col=col,
        )
    if any(v is not None for v in test):
        fig.add_trace(
            go.Scatter(x=x_vals, y=test, mode="lines+markers", name="Test"),
            row=row,
            col=col,
        )
    fig.update_xaxes(title_text=x_label, row=row, col=col)
    fig.update_yaxes(title_text="Loss / Score", row=row, col=col)


def add_estimator_summary(fig: go.Figure, estimator) -> None:
    """
    Annotate the figure with estimator parameters.

    Args:
        fig: Plotly figure to mutate.
        estimator: Estimator object exposing .to_dict().
    """
    if not hasattr(estimator, "to_dict"):
        return
    params = estimator.to_dict()
    if not isinstance(params, dict) or not params:
        return
    # Format as lightweight HTML list.
    lines = ["Estimator parameters:"] + [f"{k}: {v}" for k, v in params.items()]
    summary = "<br>".join(lines)
    fig.update_layout(
        legend=dict(
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            traceorder="normal",
        ),
    )
    line_count = max(len(lines), 1)
    summary_top = max(0.0, 1.0 - 0.1 - 0.04 * (line_count - 1))
    fig.add_annotation(
        text=summary,
        align="left",
        showarrow=False,
        xref="paper",
        yref="paper",
        x=1.02,
        y=summary_top,
        xanchor="left",
        yanchor="top",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
        borderpad=6,
        bgcolor="rgba(255,255,255,0.9)",
        font=dict(size=11),
    )
