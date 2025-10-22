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
from ..common.prediction import point_predict
from .panels import add_surfaces_2d


class ModelPerformance2DVisualizer(BaseVisualizer):
    """2D→2D: row-1 surfaces; rows 2–5 diagnostics (shared)."""

    def plot(self, data: dict) -> None:
        est = data["estimator"]
        Xtr, ytr = np.asarray(data["X_train"]), np.asarray(data["y_train"])
        Xte = np.asarray(data["X_test"]) if data.get("X_test") is not None else None
        yte = np.asarray(data["y_test"]) if data.get("y_test") is not None else None
        title = data.get("title", f"Model fit ({type(est).__name__})")
        loss_history = data["loss_history"]

        fig = make_subplots(
            rows=5,
            cols=2,
            specs=[
                [{"type": "surface"}, {"type": "surface"}],
                [{"type": "xy", "colspan": 2}, None],
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy", "colspan": 2}, None],
            ],
            vertical_spacing=0.06,
            horizontal_spacing=0.07,
            subplot_titles=[
                "Objectives (x1,x2) → Decision y1 (normalized)",
                "Objectives (x1,x2) → Decision y2 (normalized)",
                "Training / Validation / Test",
                "Residuals vs Fitted (decision y1)",
                "Residuals vs Fitted (decision y2)",
                "Residual distribution (decision y1)",
                "Residual distribution (decision y2)",
                "Residual joint distribution (y1 vs y2)",
            ],
            row_heights=[0.42, 0.14, 0.18, 0.18, 0.08],
        )

        # Row 1
        add_surfaces_2d(
            fig, row=1, estimator=est, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte
        )

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

        # Row 3: residuals vs fitted
        if yhat_te is not None:
            add_residuals_vs_fitted(
                fig,
                row=3,
                col=1,
                fitted=yhat_te[:, 0],
                resid=resid_te[:, 0],
                label="y1 (test)",
            )
            add_residuals_vs_fitted(
                fig,
                row=3,
                col=2,
                fitted=yhat_te[:, 1],
                resid=resid_te[:, 1],
                label="y2 (test)",
            )
        add_residuals_vs_fitted(
            fig,
            row=3,
            col=1,
            fitted=yhat_tr[:, 0],
            resid=resid_tr[:, 0],
            label="y1 (train)",
        )
        add_residuals_vs_fitted(
            fig,
            row=3,
            col=2,
            fitted=yhat_tr[:, 1],
            resid=resid_tr[:, 1],
            label="y2 (train)",
        )

        # Row 4: hist
        if resid_te is not None:
            add_residual_hist(
                fig, row=4, col=1, resid=resid_te[:, 0], label="y1 (test)"
            )
            add_residual_hist(
                fig, row=4, col=2, resid=resid_te[:, 1], label="y2 (test)"
            )
        add_residual_hist(fig, row=4, col=1, resid=resid_tr[:, 0], label="y1 (train)")
        add_residual_hist(fig, row=4, col=2, resid=resid_tr[:, 1], label="y2 (train)")

        # Row 5: joint (if test)
        if resid_te is not None:
            add_joint_residual(
                fig, row=5, col=1, resid_y1=resid_te[:, 0], resid_y2=resid_te[:, 1]
            )

        add_estimator_summary(fig, est)
        fig.update_layout(
            title=title + " — fit & diagnostics (normalized)",
            template="plotly_white",
            height=1600,
            width=1600,
            margin=dict(l=60, r=280, t=80, b=60),
        )
        fig.show()
