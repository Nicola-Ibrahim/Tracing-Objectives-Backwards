import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ....domain.common.interfaces.base_visualizer import BaseVisualizer
from .layout_config import (
    DEFAULT_PLOT_HEIGHT,
    DEFAULT_WIDTH,
    MAX_COLS,
    PARETO_MARKER,
    TARGET_MARKER,
    select_palette,
)


class DecisionGenerationComparisonVisualizer(BaseVisualizer):
    """Grid-based visualization comparing generated objectives across inverse models."""

    def plot(self, data: object) -> go.Figure:
        """Execute the comparison plot generation."""
        payload = self._prepare_data(data)
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

    def _prepare_data(self, data: object) -> dict:
        """Coerce and validate incoming data payload."""
        payload = self._coerce_payload(data)

        # Validate Pareto Front
        pareto_front = np.asarray(payload["pareto_front"], dtype=float)
        if pareto_front.ndim == 1:
            pareto_front = pareto_front.reshape(-1, 1)
        if pareto_front.shape[1] < 2:
            raise ValueError("Pareto front must have at least two objective columns.")
        payload["pareto_front"] = pareto_front

        # Validate Target
        target = np.asarray(payload["target_objective"], dtype=float).reshape(-1)
        if target.size < 2:
            raise ValueError("Target objective must have at least two dimensions.")
        payload["target_objective"] = target

        return payload

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

            # Common background traces
            self._add_pareto_front(fig, pareto_front, idx, row, col)
            self._add_target(fig, target, idx, row, col)

            # Model specific predictions
            color = colors[idx % len(colors)]
            self._add_prediction_trace(fig, run, idx, row, col, color)

            fig.update_xaxes(title_text="Objective 1", row=row, col=col)
            fig.update_yaxes(title_text="Objective 2", row=row, col=col)

    def _add_prediction_trace(
        self, fig: go.Figure, run: dict, idx: int, row: int, col: int, color: str
    ) -> None:
        """Add the scatter trace for model predictions."""
        name = str(run.get("name", f"model_{idx}"))
        predicted = np.asarray(run.get("predicted_objectives"), dtype=float)

        if predicted.ndim == 1:
            predicted = predicted.reshape(-1, 1)
        if predicted.shape[1] < 2:
            raise ValueError(
                f"Predicted objectives for '{name}' must have at least two columns."
            )

        fig.add_trace(
            go.Scatter(
                x=predicted[:, 0],
                y=predicted[:, 1],
                mode="markers",
                name=name,
                marker=dict(color=color, size=6, opacity=0.7),
                showlegend=True,
            ),
            row=row,
            col=col,
        )

    def _add_pareto_front(
        self, fig: go.Figure, pareto_front: np.ndarray, idx: int, row: int, col: int
    ) -> None:
        """Add the Pareto front background trace."""
        fig.add_trace(
            go.Scatter(
                x=pareto_front[:, 0],
                y=pareto_front[:, 1],
                mode="markers",
                name="Pareto Front",
                marker=PARETO_MARKER,
                showlegend=(idx == 0),
                legendgroup="pareto",
            ),
            row=row,
            col=col,
        )

    def _add_target(
        self, fig: go.Figure, target: np.ndarray, idx: int, row: int, col: int
    ) -> None:
        """Add the target objective star trace."""
        fig.add_trace(
            go.Scatter(
                x=[target[0]],
                y=[target[1]],
                mode="markers",
                name="Target Objective",
                marker=TARGET_MARKER,
                showlegend=(idx == 0),
                legendgroup="target",
            ),
            row=row,
            col=col,
        )

    def _finalize_layout(self, fig: go.Figure, n_rows: int) -> None:
        """Apply final layout styling."""
        fig.update_layout(
            title="Decision Generation Comparison",
            template="plotly_white",
            width=DEFAULT_WIDTH,
            height=DEFAULT_PLOT_HEIGHT * n_rows,
        )

    @staticmethod
    def _coerce_payload(data: object) -> dict[str, object]:
        """Convert input data into a standardized dictionary format."""
        if isinstance(data, dict):
            return data

        # Compatibility for legacy object data
        if all(
            hasattr(data, attr)
            for attr in ["pareto_front", "target_objective", "generators"]
        ):
            return {
                "pareto_front": getattr(data, "pareto_front"),
                "target_objective": getattr(data, "target_objective"),
                "generators": [
                    {
                        "name": getattr(run, "name", None),
                        "decisions": getattr(run, "decisions", None),
                        "predicted_objectives": getattr(
                            run, "predicted_objectives", None
                        ),
                    }
                    for run in getattr(data, "generators")
                ],
            }

        raise TypeError(
            "DecisionGenerationComparisonVisualizer expects a dict or object "
            "with 'pareto_front', 'target_objective', and 'generators'."
        )
