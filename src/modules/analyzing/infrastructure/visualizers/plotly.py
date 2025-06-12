import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...domain.interfaces.base_visualizer import BaseParetoVisualizer


class PlotlyParetoVisualizer(BaseParetoVisualizer):
    def plot(self, pareto_set: np.ndarray, pareto_front: np.ndarray) -> None:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[
                "Pareto Set (Decision Space)",
                "True Pareto Front (Objective Space)",
                "Overlay: Decision Space",
                "Overlay: Objective Space",
            ],
            horizontal_spacing=0.1,
            vertical_spacing=0.15,
        )

        # Pareto Set (x space)
        fig.add_trace(
            go.Scatter(
                x=pareto_set[:, 0],
                y=pareto_set[:, 1],
                mode="markers",
                name="Pareto Set",
                marker=dict(color="blue", size=6),
            ),
            row=1,
            col=1,
        )

        # Pareto Front (f space)
        fig.add_trace(
            go.Scatter(
                x=pareto_front[:, 0],
                y=pareto_front[:, 1],
                mode="markers",
                name="Pareto Front",
                marker=dict(color="green", size=6),
            ),
            row=1,
            col=2,
        )

        # Layout settings
        fig.update_layout(
            height=500,
            width=1000,
            title_text="Pareto Optimization Visualization",
            showlegend=True,
        )

        # Axis labels
        fig.update_xaxes(title_text="$x_1$", row=1, col=1)
        fig.update_yaxes(title_text="$x_2$", row=1, col=1)

        fig.update_xaxes(title_text="$f_1$", row=1, col=2)
        fig.update_yaxes(title_text="$f_2$", row=1, col=2)

        fig.show()
