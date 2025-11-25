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
from ..common.diagnostics import add_qq_plot


class ModelPerformance2DVisualizer(BaseVisualizer):
    """2D→2D: row-1 surfaces; rows 2–5 diagnostics (shared)."""

    def plot(self, data: dict) -> None:
        est = data["estimator"]
        mapping_direction = data.get("mapping_direction", "inverse")
        if mapping_direction == "inverse":
            input_symbol = "y"
            output_symbol = "x"
        else:
            input_symbol = "x"
            output_symbol = "y"

        def _sym(symbol: str, idx: int) -> str:
            subscripts = {1: "\u2081", 2: "\u2082"}
            return f"{symbol}{subscripts.get(idx, idx)}"

        Xtr, ytr = np.asarray(data["X_train"]), np.asarray(data["y_train"])
        Xte = np.asarray(data["X_test"]) if data.get("X_test") is not None else None
        yte = np.asarray(data["y_test"]) if data.get("y_test") is not None else None
        title = data.get("title", f"Model fit ({type(est).__name__})")
        loss_history = data["loss_history"]

        subplot_titles = [
            f"{_sym(output_symbol, 1)}({_sym(input_symbol, 1)}, {_sym(input_symbol, 2)})",
            f"{_sym(output_symbol, 2)}({_sym(input_symbol, 1)}, {_sym(input_symbol, 2)})",
            "Training / Validation / Test",
            f"Residuals vs Fitted ({_sym(output_symbol, 1)}) (Train)",
            f"Residuals vs Fitted ({_sym(output_symbol, 1)}) (Test)",
            f"Residuals vs Fitted ({_sym(output_symbol, 2)}) (Train)",
            f"Residuals vs Fitted ({_sym(output_symbol, 2)}) (Test)",
            f"Residual distribution ({_sym(output_symbol, 1)}) (Train)",
            f"Residual distribution ({_sym(output_symbol, 1)}) (Test)",
            f"Residual distribution ({_sym(output_symbol, 2)}) (Train)",
            f"Residual distribution ({_sym(output_symbol, 2)}) (Test)",
            f"Q-Q Plot ({_sym(output_symbol, 1)}) (Train)",
            f"Q-Q Plot ({_sym(output_symbol, 1)}) (Test)",
            f"Q-Q Plot ({_sym(output_symbol, 2)}) (Train)",
            f"Q-Q Plot ({_sym(output_symbol, 2)}) (Test)",
        ]

        fig = make_subplots(
            rows=15,
            cols=2,
            specs=[
                [{"type": "surface"}, {"type": "surface"}],
                [None, None],  # Spacer
                [{"type": "xy", "colspan": 2}, None],
                [None, None],  # Spacer
                [{"type": "xy"}, {"type": "xy"}],
                [None, None],  # Spacer
                [{"type": "xy"}, {"type": "xy"}],
                [None, None],  # Spacer
                [{"type": "xy"}, {"type": "xy"}],
                [None, None],  # Spacer
                [{"type": "xy"}, {"type": "xy"}],
                [None, None],  # Spacer
                [{"type": "xy"}, {"type": "xy"}],
                [None, None],  # Spacer
                [{"type": "xy"}, {"type": "xy"}],
            ],
            vertical_spacing=0.01,
            horizontal_spacing=0.07,
            subplot_titles=subplot_titles,
            row_heights=[0.22, 0.06, 0.12, 0.06, 0.10, 0.04, 0.10, 0.06, 0.10, 0.04, 0.10, 0.06, 0.10, 0.04, 0.10],
            column_titles=["Train", "Test"]
        )

        # Row 1
        add_surfaces_2d(
            fig,
            row=1,
            estimator=est,
            X_train=Xtr,
            y_train=ytr,
            X_test=Xte,
            y_test=yte,
            input_symbol=input_symbol,
            output_symbol=output_symbol,
        )

        # Row 3
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

        # Row 5: Residuals vs Fitted (y1)
        add_residuals_vs_fitted(
            fig,
            row=5,
            col=1,
            fitted=yhat_tr[:, 0],
            resid=resid_tr[:, 0],
            label=f"{_sym(output_symbol, 1)} (train)",
            range_y=common_range,
        )
        if yhat_te is not None:
            add_residuals_vs_fitted(
                fig,
                row=5,
                col=2,
                fitted=yhat_te[:, 0],
                resid=resid_te[:, 0],
                label=f"{_sym(output_symbol, 1)} (test)",
                range_y=common_range,
            )

        # Row 7: Residuals vs Fitted (y2)
        add_residuals_vs_fitted(
            fig,
            row=7,
            col=1,
            fitted=yhat_tr[:, 1],
            resid=resid_tr[:, 1],
            label=f"{_sym(output_symbol, 2)} (train)",
            range_y=common_range,
        )
        if yhat_te is not None:
            add_residuals_vs_fitted(
                fig,
                row=7,
                col=2,
                fitted=yhat_te[:, 1],
                resid=resid_te[:, 1],
                label=f"{_sym(output_symbol, 2)} (test)",
                range_y=common_range,
            )

        # Row 9: Hist (y1)
        add_residual_hist(
            fig,
            row=9,
            col=1,
            resid=resid_tr[:, 0],
            label=f"{_sym(output_symbol, 1)} (train)",
            range_x=common_range,
        )
        if resid_te is not None:
            add_residual_hist(
                fig,
                row=9,
                col=2,
                resid=resid_te[:, 0],
                label=f"{_sym(output_symbol, 1)} (test)",
                range_x=common_range,
            )

        # Row 11: Hist (y2)
        add_residual_hist(
            fig,
            row=11,
            col=1,
            resid=resid_tr[:, 1],
            label=f"{_sym(output_symbol, 2)} (train)",
            range_x=common_range,
        )
        if resid_te is not None:
            add_residual_hist(
                fig,
                row=11,
                col=2,
                resid=resid_te[:, 1],
                label=f"{_sym(output_symbol, 2)} (test)",
                range_x=common_range,
            )

        # Row 13: Q-Q Plot (y1)
        add_qq_plot(
            fig,
            row=13,
            col=1,
            resid=resid_tr[:, 0],
            label=f"{_sym(output_symbol, 1)} (train)",
            range_y=common_range,
        )
        if resid_te is not None:
            add_qq_plot(
                fig,
                row=13,
                col=2,
                resid=resid_te[:, 0],
                label=f"{_sym(output_symbol, 1)} (test)",
                range_y=common_range,
            )

        # Row 15: Q-Q Plot (y2)
        add_qq_plot(
            fig,
            row=15,
            col=1,
            resid=resid_tr[:, 1],
            label=f"{_sym(output_symbol, 2)} (train)",
            range_y=common_range,
        )
        if resid_te is not None:
            add_qq_plot(
                fig,
                row=15,
                col=2,
                resid=resid_te[:, 1],
                label=f"{_sym(output_symbol, 2)} (test)",
                range_y=common_range,
            )

        add_estimator_summary(fig, est, loss_history)

        # Add explanations under each row
        # Recalculated Y positions for 15 rows
        explanations = [
            (0.82, "<b>Model Surfaces</b>: Predicted decision surface (color) vs data points. <i>Goal</i>: Surface should follow data."),
            (0.70, "<b>Learning Curves</b>: Loss over epochs. <i>Goal</i>: Decrease and converge. Gap = Overfitting."),
            (0.58, "<b>Residuals vs Fitted (y1)</b>: Train (Left) vs Test (Right). <i>Goal</i>: Random scatter around 0. Similar patterns."),
            (0.50, "<b>Residuals vs Fitted (y2)</b>: Train (Left) vs Test (Right). <i>Goal</i>: Random scatter around 0. Similar patterns."),
            (0.38, "<b>Error Distribution (y1)</b>: Train (Left) vs Test (Right). <i>Goal</i>: Bell-shaped centered at 0. Similar shape."),
            (0.30, "<b>Error Distribution (y2)</b>: Train (Left) vs Test (Right). <i>Goal</i>: Bell-shaped centered at 0. Similar shape."),
            (0.18, "<b>Q-Q Plot (y1)</b>: Train (Left) vs Test (Right). <i>Goal</i>: Follow <b>Red Dashed Line</b>."),
            (0.10, "<b>Q-Q Plot (y2)</b>: Train (Left) vs Test (Right). <i>Goal</i>: Follow <b>Red Dashed Line</b>."),
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
            height=2800,
            autosize=True,
            margin=dict(l=60, r=280, t=80, b=80),
        )
        fig.show()
