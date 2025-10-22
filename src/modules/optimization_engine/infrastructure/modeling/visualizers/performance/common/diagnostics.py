import numpy as np
import plotly.graph_objects as go


def add_residuals_vs_fitted(
    fig: go.Figure, *, row: int, fitted, resid, label: str, col: int = 1
) -> None:
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
    if np.size(fitted) > 0:
        x_min, x_max = float(np.nanmin(fitted)), float(np.nanmax(fitted))
        if np.isfinite(x_min) and np.isfinite(x_max) and x_min != x_max:
            fig.add_trace(
                go.Scatter(
                    x=[x_min, x_max],
                    y=[0.0, 0.0],
                    mode="lines",
                    name="Zero residual",
                    line=dict(color="black", width=1),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
    fig.update_xaxes(title_text="Fitted (norm)", row=row, col=col)
    fig.update_yaxes(title_text="Residual (norm)", row=row, col=col)


def add_residual_hist(
    fig: go.Figure, *, row: int, resid, label: str, col: int = 1
) -> None:
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
    fig.update_xaxes(title_text="Residual (norm)", row=row, col=col)
    fig.update_yaxes(title_text="Density", row=row, col=col)


def add_joint_residual(
    fig: go.Figure, *, row: int, col: int, resid_y1, resid_y2
) -> None:
    vx = resid_y1[np.isfinite(resid_y1)]
    vy = resid_y2[np.isfinite(resid_y2)]
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
    fig: go.Figure, *, row: int, loss_history: dict | None, col: int = 1
) -> None:
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
    if not hasattr(estimator, "to_dict"):
        return
    params = estimator.to_dict()
    if not isinstance(params, dict) or not params:
        return
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
