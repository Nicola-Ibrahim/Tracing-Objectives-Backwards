import numpy as np
import plotly.graph_objects as go
import pandas as pd


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
    Plot residuals vs fitted values with enhanced styling.
    """
    fitted_limits = _pad_range(_percentile_limits(fitted))
    resid_limits = _pad_range(_symmetric_limits(resid))

    # Scatter each residual vs its fitted value.
    fig.add_trace(
        go.Scatter(
            x=fitted,
            y=resid,
            mode="markers",
            name=f"Residuals ({label})",
            marker=dict(
                opacity=0.6,
                size=5,
                color=resid,
                colorscale="RdBu",
                cmid=0,
                showscale=False,
                line=dict(width=0.5, color="DarkSlateGrey"),
            ),
            hovertemplate="<b>Fitted</b>: %{x:.4f}<br><b>Resid</b>: %{y:.4f}<extra></extra>",
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
            line=dict(color="black", width=1.5, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )
    # Apply the normalized ranges to both axes.
    fig.update_xaxes(
        title_text="Fitted (norm)",
        range=list(fitted_limits),
        row=row,
        col=col,
        gridcolor="lightgrey",
    )
    fig.update_yaxes(
        title_text="Residual (norm)",
        range=list(resid_limits),
        row=row,
        col=col,
        gridcolor="lightgrey",
        zeroline=True,
        zerolinecolor="grey",
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
    Plot a residual histogram with density scaling and KDE-like look.
    """
    resid_limits = _pad_range(_symmetric_limits(resid))
    fig.add_trace(
        go.Histogram(
            x=resid,
            nbinsx=50,
            histnorm="probability density",
            name=f"Resid ({label})",
            opacity=0.7,
            marker=dict(color="Teal", line=dict(width=0.5, color="white")),
            hovertemplate="<b>Resid</b>: %{x:.4f}<br><b>Density</b>: %{y:.4f}<extra></extra>",
        ),
        row=row,
        col=col,
    )
    # Highlight zero
    vals = resid[np.isfinite(resid)]
    if vals.size:
        counts, _ = np.histogram(vals, bins=50, density=True)
        ymax = float(np.nanmax(counts)) if counts.size else 1.0
        fig.add_trace(
            go.Scatter(
                x=[0.0, 0.0],
                y=[0.0, ymax * 1.1],
                mode="lines",
                name="Zero",
                line=dict(color="black", width=1.5, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
    fig.update_xaxes(
        title_text="Residual (norm)",
        range=list(resid_limits),
        row=row,
        col=col,
        gridcolor="lightgrey",
    )
    fig.update_yaxes(title_text="Density", row=row, col=col, gridcolor="lightgrey")


def add_joint_residual(
    fig: go.Figure,
    *,
    row: int,
    col: int,
    resid_y1,
    resid_y2,
) -> None:
    """
    Plot a joint density of two residual dimensions with improved contours.
    """
    vx = resid_y1[np.isfinite(resid_y1)]
    vy = resid_y2[np.isfinite(resid_y2)]
    
    fig.add_trace(
        go.Histogram2dContour(
            x=vx,
            y=vy,
            ncontours=25,
            colorscale="Viridis",
            showscale=False,
            name="Residual density",
            opacity=0.8,
            hovertemplate="<b>Resid 1</b>: %{x:.4f}<br><b>Resid 2</b>: %{y:.4f}<br><b>Count</b>: %{z}<extra></extra>",
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=vx,
            y=vy,
            mode="markers",
            name="Residuals",
            marker=dict(size=3, opacity=0.3, color="black"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )
    fig.update_xaxes(title_text="Residual y1 (norm)", row=row, col=col, gridcolor="lightgrey")
    fig.update_yaxes(title_text="Residual y2 (norm)", row=row, col=col, gridcolor="lightgrey")


def add_loss_curves(
    fig: go.Figure,
    *,
    row: int,
    loss_history: dict | None,
    col: int = 1,
) -> None:
    """
    Draw training/validation/test loss curves with better styling.
    """
    if loss_history is None:
        return
    lh = loss_history
    if hasattr(lh, "model_dump"):
        lh = lh.model_dump()
    elif hasattr(lh, "dict"):
        lh = lh.dict()
    if not isinstance(lh, dict):
        return

    bins = list(lh.get("bins", []))
    train = list(lh.get("train_loss", []))
    val = list(lh.get("val_loss", []))
    test = list(lh.get("test_loss", []))
    n_tr = list(lh.get("n_train", []))
    bin_type: str = str(lh.get("bin_type", "bin"))

    if bin_type == "train_fraction" and n_tr:
        x_vals, x_label = n_tr, "Number of Training Samples"
    elif bin_type == "epoch":
        x_vals, x_label = (bins if bins else list(range(1, len(train) + 1))), "Epoch"
    else:
        x_vals = bins if bins else list(range(len(train)))
        x_label = {"param": "Parameter", "bin": "Bin"}.get(bin_type, bin_type.title())

    if train:
        fig.add_trace(
            go.Scatter(
                x=x_vals, 
                y=train, 
                mode="lines", 
                name="Train Loss",
                line=dict(color="RoyalBlue", width=2),
                hovertemplate="<b>Train</b>: %{y:.4f}<extra></extra>"
            ),
            row=row,
            col=col,
        )
    if any(v is not None for v in val):
        fig.add_trace(
            go.Scatter(
                x=x_vals, 
                y=val, 
                mode="lines", 
                name="Val Loss",
                line=dict(color="FireBrick", width=2),
                hovertemplate="<b>Val</b>: %{y:.4f}<extra></extra>"
            ),
            row=row,
            col=col,
        )
    if any(v is not None for v in test):
        fig.add_trace(
            go.Scatter(
                x=x_vals, 
                y=test, 
                mode="lines", 
                name="Test Loss",
                line=dict(color="ForestGreen", width=2, dash="dot"),
                hovertemplate="<b>Test</b>: %{y:.4f}<extra></extra>"
            ),
            row=row,
            col=col,
        )
    fig.update_xaxes(title_text=x_label, row=row, col=col, gridcolor="lightgrey")
    fig.update_yaxes(title_text="Loss / Score", row=row, col=col, gridcolor="lightgrey")


def add_estimator_summary(
    fig: go.Figure, 
    estimator, 
    loss_history: dict | None = None
) -> None:
    """
    Annotate the figure with estimator parameters and final loss statistics.

    Args:
        fig: Plotly figure to mutate.
        estimator: Estimator object exposing .to_dict().
        loss_history: Optional loss history dictionary.
    """
    lines = []
    
    # Add Estimator Parameters
    if hasattr(estimator, "to_dict"):
        params = estimator.to_dict()
        if isinstance(params, dict) and params:
            lines.append("<b>Estimator parameters:</b>")
            lines.extend([f"{k}: {v}" for k, v in params.items()])
            lines.append("<br>")  # Spacer

    # Add Loss Statistics
    if loss_history:
        lh = loss_history
        if hasattr(lh, "model_dump"):
            lh = lh.model_dump()
        elif hasattr(lh, "dict"):
            lh = lh.dict()
            
        if isinstance(lh, dict):
            lines.append("<b>Final Loss Statistics:</b>")
            
            # Helper to get last valid value
            def get_last(key):
                vals = lh.get(key, [])
                valid = [v for v in vals if v is not None]
                return valid[-1] if valid else None

            train_loss = get_last("train_loss")
            val_loss = get_last("val_loss")
            test_loss = get_last("test_loss")

            if train_loss is not None:
                lines.append(f"Train Loss: {train_loss:.4f}")
            if val_loss is not None:
                lines.append(f"Val Loss: {val_loss:.4f}")
            if test_loss is not None:
                lines.append(f"Test Loss: {test_loss:.4f}")

    if not lines:
        return

    summary = "<br>".join(lines)
    
    # Update layout to make room for the annotation if needed
    # (Though usually the right margin is sufficient)
    
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
    
    # Calculate dynamic position based on content length
    # But fixed top-right is usually best for summary
    fig.add_annotation(
        text=summary,
        align="left",
        showarrow=False,
        xref="paper",
        yref="paper",
        x=1.02,
        y=0.95,  # Slightly below top to avoid overlapping with potential title if wide
        xanchor="left",
        yanchor="top",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
        borderpad=6,
        bgcolor="rgba(255,255,255,0.9)",
        font=dict(size=11),
    )


def add_qq_plot(
    fig: go.Figure,
    *,
    row: int,
    resid,
    label: str,
    col: int = 1,
) -> None:
    """
    Add a Q-Q plot to check for normality of residuals.
    """
    from scipy import stats

    vals = _finite(resid)
    if vals.size < 2:
        return

    (osm, osr), (slope, intercept, r) = stats.probplot(vals, dist="norm", fit=True)

    fig.add_trace(
        go.Scatter(
            x=osm,
            y=osr,
            mode="markers",
            name=f"Q-Q ({label})",
            marker=dict(size=5, opacity=0.6, color="RoyalBlue"),
            hovertemplate="<b>Theoretical</b>: %{x:.4f}<br><b>Sample</b>: %{y:.4f}<extra></extra>",
        ),
        row=row,
        col=col,
    )

    x_line = np.array([osm.min(), osm.max()])
    y_line = slope * x_line + intercept
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="Normal Fit",
            line=dict(color="Red", width=1.5, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )

    fig.update_xaxes(title_text="Theoretical Quantiles", row=row, col=col, gridcolor="lightgrey")
    fig.update_yaxes(title_text="Sample Quantiles", row=row, col=col, gridcolor="lightgrey")
