import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ....shared.domain.interfaces.base_visualizer import BaseVisualizer
from .panels.layout_config import (
    DEFAULT_WIDTH,
    select_palette,
)
from .panels.payload_handler import prepare_payload
from .panels.traces import (
    add_best_decision_trace,
    add_decision_trace,
    add_objective_connection_trace,
    add_pareto_front_trace,
    add_pareto_set_trace,
    add_prediction_trace,
    add_target_trace,
)


class DecisionGenerationComparisonVisualizer(BaseVisualizer):
    """Grid-based visualization comparing generated objectives across inverse models."""

    def __init__(self):
        super().__init__()
        self._current_dataset_name: str | None = None

    def _initialize_figure(self, generators: list[dict]) -> go.Figure:
        """Initialize the subplot grid with row-based titles."""
        n_rows = len(generators)
        titles = []
        for run in generators:
            name = run.get("name", "Model")
            titles.extend([f"{name} (Decisions)", f"{name} (Objectives)"])

        return make_subplots(
            rows=n_rows,
            cols=2,
            subplot_titles=titles,
            horizontal_spacing=0.12,
            vertical_spacing=max(0.01, 0.4 / n_rows) if n_rows > 1 else 0.1,
        )

    def _populate_subplots(self, fig: go.Figure, payload: dict) -> None:
        """Add traces for each model generator to the figure."""
        generators = payload["generators"]
        pareto_front = payload["pareto_front"]
        pareto_set = payload.get("pareto_set")
        target = payload["target_objective"]
        colors = select_palette(len(generators))

        for idx, run in enumerate(generators):
            row = idx + 1
            color = colors[idx % len(colors)]

            # Column 1: Decision Space
            show_legend = idx == 0
            if pareto_set is not None:
                add_pareto_set_trace(fig, pareto_set, show_legend, row, 1)

            add_decision_trace(fig, run, idx, row, 1, color)
            add_best_decision_trace(fig, run, row, 1, color)
            fig.update_xaxes(title_text="x1", row=row, col=1)
            fig.update_yaxes(title_text="x2", row=row, col=1)

            # Column 2: Objective Space
            show_legend = idx == 0
            add_pareto_front_trace(fig, pareto_front, show_legend, row, 2)
            add_target_trace(fig, target, show_legend, row, 2)
            add_prediction_trace(fig, run, idx, row, 2, color)
            add_objective_connection_trace(fig, run, target, row, 2, color)

            fig.update_xaxes(title_text="y1", row=row, col=2)
            fig.update_yaxes(title_text="y2", row=row, col=2)

    def _finalize_layout(self, fig: go.Figure, n_rows: int) -> None:
        """Apply final layout styling."""
        dataset_name = self._current_dataset_name
        fig.update_layout(
            title=self._format_title(dataset_name),
            template="plotly_white",
            width=DEFAULT_WIDTH,
            height=max(800, 450 * n_rows),
            margin=dict(t=120, b=80, l=80, r=80),
        )

    def _format_title(self, dataset_name: str | None) -> str:
        base = "Decision Generation Comparison"
        return f"{base} — {dataset_name}" if dataset_name else base

    def plot(self, data: object) -> go.Figure:
        """Execute the comparison plot generation."""
        payload = prepare_payload(data)
        self._current_dataset_name = payload.get("dataset_name")
        generators = payload["generators"]

        if not generators:
            raise ValueError("No generator results provided for visualization.")

        n_rows = len(generators)

        fig = self._initialize_figure(generators)
        self._populate_subplots(fig, payload)
        self._finalize_layout(fig, n_rows)

        fig.show()
        return fig
