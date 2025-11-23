import numpy as np
from plotly.subplots import make_subplots

from ......domain.visualization.interfaces.base_visualizer import BaseVisualizer
from ..common.diagnostics import (
    add_estimator_summary,
    add_joint_residual,
    add_loss_curves,
    add_residual_hist,
    add_residuals_vs_fitted,
)
from ..common.prediction import point_predict, sample_band
from .panels import add_fit_1d, add_points_overlay
from .reduction import reduce_to_1d, transform_1d


class ModelPerformanceNDVisualizer(BaseVisualizer):
    """General case (not 2D→2D): row-1 1D fit + ribbon; rows 2–5 diagnostics."""

    def plot(self, data: dict) -> None:
        est = data["estimator"]
        Xtr, ytr = np.asarray(data["X_train"]), np.asarray(data["y_train"])
        Xte = np.asarray(data["X_test"]) if data.get("X_test") is not None else None
        yte = np.asarray(data["y_test"]) if data.get("y_test") is not None else None
        n_samples = int(data.get("n_samples", 200))
        title = data.get("title", f"Model fit ({type(est).__name__})")
        loss_history = data["loss_history"]

        fig = make_subplots(
            rows=6,
            cols=1,
            specs=[
                [{"type": "xy"}],
                [{"type": "xy"}],
                [{"type": "xy"}],
                [{"type": "xy"}],
                [{"type": "xy"}],
                [{"type": "xy"}],
            ],
            vertical_spacing=0.08,
            subplot_titles=(
                f"{title} — y0 (normalized)",
                "Training / Validation / Test",
                "Residuals vs Fitted (train + test)",
                "Residual distribution (train + test)",
                "Residual joint distribution",
                "Q-Q Plot",
            ),
            row_heights=[0.2, 0.1, 0.1, 0.1, 0.15, 0.15],
        )

        # Row 1: reduce, center+band, overlay
        Xr_train, reducer = reduce_to_1d(Xtr)
        Xr_test = transform_1d(reducer, Xte)
        center, p05, p95 = sample_band(est, Xtr, n_samples)
        add_fit_1d(
            fig,
            row=1,
            X_red=Xr_train,
            center=center,
            p05=p05,
            p95=p95,
            name_center=("MAP" if hasattr(est, "predict_map") else "Prediction"),
        )
        add_points_overlay(
            fig,
            row=1,
            X_train_red=Xr_train,
            y_train_1d=ytr[:, 0:1],
            X_test_red=Xr_test,
            y_test_1d=(yte[:, 0:1] if yte is not None else None),
        )
        fig.update_xaxes(title_text="Objective (reduced, normalized)", row=1, col=1)
        fig.update_yaxes(title_text="Decision (normalized)", row=1, col=1)

        # Row 2
        add_loss_curves(fig, row=2, loss_history=loss_history, col=1)

        # Residual diagnostics
        yhat_tr = point_predict(est, Xtr)
        resid_tr = ytr - yhat_tr
        if Xte is not None and yte is not None:
            yhat_te = point_predict(est, Xte)
            resid_te = yte - yhat_te
        else:
            yhat_te = resid_te = None

        # Row 3
        if yhat_te is not None:
            add_residuals_vs_fitted(
                fig,
                row=3,
                fitted=yhat_te[:, 0],
                resid=resid_te[:, 0],
                label="y0 (test)",
            )
        add_residuals_vs_fitted(
            fig, row=3, fitted=yhat_tr[:, 0], resid=resid_tr[:, 0], label="y0 (train)"
        )

        # Row 4
        if resid_te is not None:
            add_residual_hist(fig, row=4, resid=resid_te[:, 0], label="y0 (test)")
        add_residual_hist(fig, row=4, resid=resid_tr[:, 0], label="y0 (train)")

        # Row 5
        if resid_te is not None and ytr.shape[1] >= 2:
            add_joint_residual(
                fig, row=5, col=1, resid_y1=resid_te[:, 0], resid_y2=resid_te[:, 1]
            )

        # Row 6: Q-Q Plot
        from ..common.diagnostics import add_qq_plot

        if resid_te is not None:
            add_qq_plot(fig, row=6, resid=resid_te[:, 0], label="y0 (test)")
        add_qq_plot(fig, row=6, resid=resid_tr[:, 0], label="y0 (train)")

        add_estimator_summary(fig, est, loss_history)
        fig.update_layout(
            title=title + " — fit & diagnostics (normalized)",
            template="plotly_white",
            height=1600,
            width=1000,
            margin=dict(l=60, r=280, t=80, b=60),
        )
        fig.show()
