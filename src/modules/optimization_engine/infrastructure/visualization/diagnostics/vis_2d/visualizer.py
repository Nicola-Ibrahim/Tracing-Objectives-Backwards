import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .....domain.common.interfaces.base_visualizer import BaseVisualizer
from ..common.diagnostics import (
    add_estimator_summary,
    add_loss_curves,
)
from .layout_config import DIAGNOSTIC_EXPLANATIONS, get_subplot_titles
from .panels import add_surfaces_2d


class ModelPerformance2DVisualizer(BaseVisualizer):
    """2D→2D Model Diagnostic Visualizer.

    Layout:
    - Row 1: Predicted surfaces (o1, o2) vs input space (s1, s2).
    - Row 2: Spacer.
    - Row 3: Learning curves (Loss history).
    """

    def plot(self, data: dict) -> go.Figure:
        """Generate and display the diagnostic plot."""
        processed_data = self._prepare_data(data)
        fig = self._initialize_figure(processed_data)

        self._add_diagnostic_panels(fig, processed_data)
        self._add_annotations(fig)
        self._finalize_layout(fig, processed_data)

        fig.show()
        return fig

    def _prepare_data(self, data: dict) -> dict:
        """Extract and format data for visualization."""
        mapping_direction = data.get("mapping_direction", "inverse")
        input_symbol = "x" if mapping_direction == "forward" else "y"
        output_symbol = "y" if mapping_direction == "forward" else "x"

        return {
            "estimator": data["estimator"],
            "X_train": np.asarray(data["X_train"]),
            "y_train": np.asarray(data["y_train"]),
            "X_test": (
                np.asarray(data["X_test"]) if data.get("X_test") is not None else None
            ),
            "y_test": (
                np.asarray(data["y_test"]) if data.get("y_test") is not None else None
            ),
            "training_history": data["training_history"],
            "title": data.get(
                "title", f"Model fit ({type(data['estimator']).__name__})"
            ),
            "input_symbol": input_symbol,
            "output_symbol": output_symbol,
        }

    def _initialize_figure(self, data: dict) -> go.Figure:
        """Initialize plotly subplots with the correct specification."""
        subplot_titles = get_subplot_titles(data["input_symbol"], data["output_symbol"])

        return make_subplots(
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

    def _add_diagnostic_panels(self, fig: go.Figure, data: dict) -> None:
        """Populate the figure with diagnostic plots."""
        # Row 1: Surfaces
        add_surfaces_2d(
            fig,
            row=1,
            estimator=data["estimator"],
            X_train=data["X_train"],
            y_train=data["y_train"],
            X_test=data["X_test"],
            y_test=data["y_test"],
            input_symbol=data["input_symbol"],
            output_symbol=data["output_symbol"],
        )

        # Row 3: Loss curves
        add_loss_curves(fig, row=3, training_history=data["training_history"], col=1)

        # Sidebar/Global: Estimator summary
        add_estimator_summary(fig, data["estimator"], data["training_history"])

    def _add_annotations(self, fig: go.Figure) -> None:
        """Add explanatory notes to the visualization."""
        for y_pos, text in DIAGNOSTIC_EXPLANATIONS:
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

    def _finalize_layout(self, fig: go.Figure, data: dict) -> None:
        """Fine-tune final layout parameters."""
        full_title = f"{data['title']} — probabilistic model visualization (normalized)"
        fig.update_layout(
            title=full_title,
            template="plotly_white",
            height=1200,
            autosize=True,
            margin=dict(l=60, r=280, t=80, b=80),
        )
