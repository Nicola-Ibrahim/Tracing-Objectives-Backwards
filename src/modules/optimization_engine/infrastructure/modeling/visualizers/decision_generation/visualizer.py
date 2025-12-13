import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .....domain.visualization.interfaces.base_visualizer import BaseVisualizer


class DecisionGenerationComparisonVisualizer(BaseVisualizer):
    """Grid-based visualization comparing generated objectives across inverse models."""

    _MAX_COLS = 3

    def plot(self, data: object) -> None:
        payload = self._coerce_payload(data)
        generators = payload["generators"]
        if not generators:
            raise ValueError("No generator results provided for visualization.")

        n_models = len(generators)
        n_cols = min(self._MAX_COLS, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        pareto_front = np.asarray(payload["pareto_front"], dtype=float)
        if pareto_front.ndim == 1:
            pareto_front = pareto_front.reshape(-1, 1)
        if pareto_front.shape[1] < 2:
            raise ValueError("Pareto front must have at least two objective columns.")

        target = np.asarray(payload["target_objective"], dtype=float).reshape(-1)
        if target.size < 2:
            raise ValueError("Target objective must have at least two dimensions.")

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[
                f"{str(run.get('name', f'model_{idx}'))} Predictions"
                for idx, run in enumerate(generators)
            ],
        )

        colors = self._select_palette(n_models)

        for idx, run in enumerate(generators):
            row, col = divmod(idx, n_cols)
            row += 1
            col += 1
            name = str(run.get("name", f"model_{idx}"))
            predicted = np.asarray(run.get("predicted_objectives"), dtype=float)
            if predicted.ndim == 1:
                predicted = predicted.reshape(-1, 1)
            if predicted.shape[1] < 2:
                raise ValueError(
                    f"Predicted objectives for '{name}' must have at least two columns."
                )
            color = colors[idx % len(colors)]

            self._add_pareto_front(fig, pareto_front, idx, row, col)
            self._add_target(fig, target, idx, row, col)

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

            fig.update_xaxes(title_text="Objective 1", row=row, col=col)
            fig.update_yaxes(title_text="Objective 2", row=row, col=col)

        fig.update_layout(
            title="Decision Generation Comparison",
            template="plotly_white",
            width=1800,
            height=600 * n_rows,
        )

        fig.show()

    @staticmethod
    def _coerce_payload(data: object) -> dict[str, object]:
        if isinstance(data, dict):
            return data
        # Backward-compatibility for legacy DecisionGenerationVisualizationData
        if hasattr(data, "pareto_front") and hasattr(data, "target_objective") and hasattr(
            data, "generators"
        ):
            generators = []
            for run in list(getattr(data, "generators")):
                generators.append(
                    {
                        "name": getattr(run, "name", None),
                        "decisions": getattr(run, "decisions", None),
                        "predicted_objectives": getattr(run, "predicted_objectives", None),
                    }
                )
            return {
                "pareto_front": getattr(data, "pareto_front"),
                "target_objective": getattr(data, "target_objective"),
                "generators": generators,
            }
        raise TypeError(
            "DecisionGenerationComparisonVisualizer.plot expects a dict with "
            "keys: 'pareto_front', 'target_objective', 'generators'."
        )

    def _add_pareto_front(
        self, fig: go.Figure, pareto_front: np.ndarray, idx: int, row: int, col: int
    ) -> None:
        fig.add_trace(
            go.Scatter(
                x=pareto_front[:, 0],
                y=pareto_front[:, 1],
                mode="markers",
                name="Pareto Front",
                marker=dict(color="lightgray", size=5, opacity=0.3),
                showlegend=(idx == 0),
                legendgroup="pareto",
            ),
            row=row,
            col=col,
        )

    def _add_target(
        self, fig: go.Figure, target: np.ndarray, idx: int, row: int, col: int
    ) -> None:
        fig.add_trace(
            go.Scatter(
                x=[target[0]],
                y=[target[1]],
                mode="markers",
                name="Target Objective",
                marker=dict(
                    color="red",
                    symbol="star",
                    size=15,
                    line=dict(width=2, color="black"),
                ),
                showlegend=(idx == 0),
                legendgroup="target",
            ),
            row=row,
            col=col,
        )

    @staticmethod
    def _select_palette(n_models: int) -> list[str]:
        if n_models <= 10:
            return px.colors.qualitative.Plotly
        if n_models <= 24:
            return px.colors.qualitative.Dark24
        return px.colors.qualitative.Dark24 + px.colors.qualitative.Plotly
