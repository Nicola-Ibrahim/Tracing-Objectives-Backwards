import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ....domain.common.interfaces.base_visualizer import BaseVisualizer
from .panels.layout_config import (
    DEFAULT_PLOT_HEIGHT,
    DEFAULT_WIDTH,
    MAX_COLS,
    select_palette,
)
from .panels.payload_handler import prepare_payload
from .panels.traces import add_pareto_front_trace, add_prediction_trace, add_target_trace


class DecisionGenerationComparisonVisualizer(BaseVisualizer):
    """Grid-based visualization comparing generated objectives across inverse models."""

    def plot(self, data: object) -> go.Figure:
        """Execute the comparison plot generation."""
        payload = prepare_payload(data)
        generators = payload["generators"]

        if not generators:
            raise ValueError("No generator results provided for visualization.")

        n_models = len(generators)
        n_cols = min(MAX_COLS, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig = self._initialize_figure(n_rows, n_cols, generators)
        self._populate_subplots(fig, payload, n_cols)
        self._finalize_layout(fig, n_rows)

        fig.show()
        return fig

    def _initialize_figure(
        self, n_rows: int, n_cols: int, generators: list[dict]
    ) -> go.Figure:
        """Initialize the subplot grid."""
        titles = [
            f"{str(run.get('name', f'model_{idx}'))} Predictions"
            for idx, run in enumerate(generators)
        ]
        return make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=titles,
        )

    def _populate_subplots(self, fig: go.Figure, payload: dict, n_cols: int) -> None:
        """Add traces for each model generator to the figure."""
        generators = payload["generators"]
        pareto_front = payload["pareto_front"]
        target = payload["target_objective"]
        colors = select_palette(len(generators))

        for idx, run in enumerate(generators):
            row, col = divmod(idx, n_cols)
            row += 1
            col += 1

            # Common background traces (show legend only for the first plot)
            show_legend = idx == 0
            add_pareto_front_trace(fig, pareto_front, show_legend, row, col)
            add_target_trace(fig, target, show_legend, row, col)

            # Model specific predictions
            color = colors[idx % len(colors)]
            add_prediction_trace(fig, run, idx, row, col, color)

            fig.update_xaxes(title_text="Objective 1", row=row, col=col)
            fig.update_yaxes(title_text="Objective 2", row=row, col=col)

    def _finalize_layout(self, fig: go.Figure, n_rows: int) -> None:
        """Apply final layout styling."""
        fig.update_layout(
            title="Decision Generation Comparison",
            template="plotly_white",
            width=DEFAULT_WIDTH,
            height=DEFAULT_PLOT_HEIGHT * n_rows,
        )
