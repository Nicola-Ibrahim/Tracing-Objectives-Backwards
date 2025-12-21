import numpy as np
from plotly.subplots import make_subplots

from ......domain.visualization.interfaces.base_visualizer import BaseVisualizer
from ..common.diagnostics import (
    add_estimator_summary,
    add_loss_curves,
)
from .panels import add_surfaces_2d


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
        ]

        fig = make_subplots(
            rows=3,
            cols=2,
            specs=[
                [{"type": "surface"}, {"type": "surface"}],
                [None, None],  # Spacer
                [{"type": "xy", "colspan": 2}, None],
            ],
            vertical_spacing=0.05,
            horizontal_spacing=0.07,
            subplot_titles=subplot_titles,
            row_heights=[0.65, 0.05, 0.30],
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

        add_estimator_summary(fig, est, loss_history)

        # Add explanations under each row
        explanations = [
            (
                0.75,
                "<b>Model Surfaces</b>: Predicted decision surface (ghost scatter for probabilistic models) vs data points. <i>Goal</i>: Scatter cloud should cover data distribution.",
            ),
            (
                0.20,
                "<b>Learning Curves</b>: Loss over epochs. <i>Goal</i>: Decrease and converge. Gap = Overfitting.",
            ),
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
            title=title + " — probabilistic model visualization (normalized)",
            template="plotly_white",
            height=1200,
            autosize=True,
            margin=dict(l=60, r=280, t=80, b=80),
        )
        fig.show()
