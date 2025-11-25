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
            rows=5,
            cols=2,
            specs=[
                [{"type": "xy", "colspan": 2}, None],
                [None, None],  # Spacer
                [{"type": "xy", "colspan": 2}, None],
                [None, None],  # Spacer
                [{"type": "xy"}, {"type": "xy"}],
                [None, None],  # Spacer
                [{"type": "xy"}, {"type": "xy"}],
                [None, None],  # Spacer
                [{"type": "xy"}, {"type": "xy"}],
            ],
            vertical_spacing=0.01,
            subplot_titles=subplot_titles,
            row_heights=[0.22, 0.06, 0.12, 0.06, 0.15, 0.06, 0.15, 0.06, 0.15],
            column_titles=["Train", "Test"]
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

        # Row 3 (was 2)
        add_loss_curves(fig, row=3, loss_history=loss_history, col=1)

        # Residual diagnostics
        yhat_tr = point_predict(est, Xtr)
        resid_tr = ytr - yhat_tr
        if Xte is not None and yte is not None:
            yhat_te = point_predict(est, Xte)
            resid_te = yte - yhat_te
        else:
            yhat_te = resid_te = None

        # Compute global residual limits for consistent scaling
        all_resids = [resid_tr]
        if resid_te is not None:
            all_resids.append(resid_te)
        concatenated = np.concatenate(all_resids, axis=0)
        # Filter finite values
        concatenated = concatenated[np.isfinite(concatenated)]
        if concatenated.size > 0:
            r_min, r_max = concatenated.min(), concatenated.max()
            span = r_max - r_min
            pad = span * 0.05 if span > 0 else 1.0
            common_range = [r_min - pad, r_max + pad]
        else:
            common_range = [-1.0, 1.0]

        # Row 5 (was 3): Residuals vs Fitted - Train (Col 1), Test (Col 2)
        add_residuals_vs_fitted(
            fig,
            row=5,
            col=1,
            fitted=yhat_tr[:, 0],
            resid=resid_tr[:, 0],
            label="y0 (train)",
            range_y=common_range,
        )
        if yhat_te is not None:
            add_residuals_vs_fitted(
                fig,
                row=5,
                col=2,
                fitted=yhat_te[:, 0],
                resid=resid_te[:, 0],
                label="y0 (test)",
                range_y=common_range,
            )

        # Row 7 (was 4): Hist - Train (Col 1), Test (Col 2)
        add_residual_hist(
            fig,
            row=7,
            col=1,
            resid=resid_tr[:, 0],
            label="y0 (train)",
            range_x=common_range,
        )
        if resid_te is not None:
            add_residual_hist(
                fig,
                row=7,
                col=2,
                resid=resid_te[:, 0],
                label="y0 (test)",
                range_x=common_range,
            )

        # Row 9 (was 5): Q-Q Plot - Train (Col 1), Test (Col 2)
        from ..common.diagnostics import add_qq_plot
        add_qq_plot(
            fig,
            row=9,
            col=1,
            resid=resid_tr[:, 0],
            label="y0 (train)",
            range_y=common_range,
        )
        if resid_te is not None:
            add_qq_plot(
                fig,
                row=9,
                col=2,
                resid=resid_te[:, 0],
                label="y0 (test)",
                range_y=common_range,
            )

        add_estimator_summary(fig, est, loss_history)

        # Add explanations under each row
        # Recalculated Y positions for 9 rows
        # Height ~2000
        # Row 1 (Fit): Top ~1.0, Bottom ~0.78. Expl ~0.80
        # Row 3 (Curves): Top ~0.75, Bottom ~0.63. Expl ~0.65
        # Row 5 (Resid): Top ~0.60, Bottom ~0.45. Expl ~0.47
        # Row 7 (Hist): Top ~0.42, Bottom ~0.27. Expl ~0.29
        # Row 9 (QQ): Top ~0.24, Bottom ~0.09. Expl ~0.11

        explanations = [
            (0.80, "<b>1D Fit</b>: Reduced 1D view of model fit with uncertainty bands. <i>Goal</i>: Blue line should follow data trend."),
            (0.65, "<b>Learning Curves</b>: Tracks loss over epochs. <i>Goal</i>: Both should decrease and converge. Large gap = Overfitting."),
            (0.47, "<b>Residuals vs Fitted</b>: Train (Left) vs Test (Right). <i>Goal</i>: Random scatter around 0. Patterns indicate non-linearity."),
            (0.29, "<b>Error Distribution</b>: Train (Left) vs Test (Right). <i>Goal</i>: Bell-shaped (Gaussian) centered at 0."),
            (0.11, "<b>Q-Q Plot</b>: Train (Left) vs Test (Right). <i>Goal</i>: Points should follow the <b>Red Dashed Line</b>."),
        ]
        for y_pos, text in explanations:
            fig.add_annotation(
                text=text,
                xref="paper",
                yref="paper",
                x=0.5,
                y=y_pos,
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                font=dict(size=11, color="#444"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1,
                borderpad=4,
            )

        fig.update_layout(
            title=title + " — fit & diagnostics (normalized)",
            template="plotly_white",
            height=2000,
            autosize=True,
            margin=dict(l=60, r=280, t=80, b=80),
        )
        fig.show()
