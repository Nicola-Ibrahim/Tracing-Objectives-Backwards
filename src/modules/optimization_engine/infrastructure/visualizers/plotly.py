import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import MinMaxScaler

from ...domain.analyzing.interfaces.base_visualizer import BaseParetoVisualizer


class PlotlyParetoVisualizer(BaseParetoVisualizer):
    """Dashboard for visualizing Pareto set and front using Plotly."""

    def plot(self, pareto_set: np.ndarray, pareto_front: np.ndarray) -> None:
        """Generate an interactive dashboard with multiple Pareto visualizations."""
        fig = make_subplots(
            rows=4,
            cols=3,
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "parcoords", "colspan": 2}, None, {"type": "scatter"}],
            ],
            subplot_titles=[
                "Decision Space ($x_1$ vs $x_2$)",
                "Objective Space ($f_1$ vs $f_2$)",
                "Decision vs Objective",
                "Normalized Decision Space",
                "Normalized Objective Space",
                "Normalized Decision vs Objective",
                "$f_1$ vs $f_2$",
                "$f_1$ vs $x_1$",
                "$f_1$ vs $x_2$",
                "Parallel Coordinates",
            ],
            horizontal_spacing=0.05,
            vertical_spacing=0.07,
        )

        fig.update_layout(
            title_text="Pareto Optimization Analysis Dashboard",
            height=1800,
            width=1600,
            showlegend=True,
            template="plotly_white",
        )

        self._add_decision_objective_spaces(fig, pareto_set, pareto_front)
        self._add_normalized_spaces(fig, pareto_set, pareto_front)
        self._add_f1_relationships(fig, pareto_set, pareto_front)
        self._add_parallel_coordinates(fig, pareto_set, pareto_front)

        fig.write_image(
            file=self.save_path / "pareto_dashboard.png",
            width=1600,
            height=1800,
            scale=2,
            engine="kaleido",
        )
        fig.show()

    def _set_axis_limits(self, fig, row, col, x_data, y_data, padding=0.05):
        """Helper to set axis limits with some padding."""
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)
        x_range = x_max - x_min
        y_range = y_max - y_min

        fig.update_xaxes(
            range=[x_min - padding * x_range, x_max + padding * x_range],
            row=row,
            col=col,
        )
        fig.update_yaxes(
            range=[y_min - padding * y_range, y_max + padding * y_range],
            row=row,
            col=col,
        )

    def _add_description(self, fig, row, col, text):
        """Add description text annotation below a subplot."""
        # Calculate axis number: col + (row - 1) * 3
        axis_num = col + (row - 1) * 3

        # For xref and yref, use "x domain" for 1, else "x{n} domain"
        if axis_num == 1:
            xref = "x domain"
            yref = "y domain"
        else:
            xref = f"x{axis_num} domain"
            yref = f"y{axis_num} domain"

        fig.add_annotation(
            text=text,
            x=0.5,
            y=-0.20,
            xref=xref,
            yref=yref,
            showarrow=False,
            font=dict(size=11, color="grey"),
            align="center",
        )

    def _add_decision_objective_spaces(self, fig, pareto_set, pareto_front):
        """Visualize Pareto set and front in original space."""
        markers = dict(size=6, opacity=0.7)

        # Decision space
        fig.add_trace(
            go.Scatter(
                x=pareto_set[:, 0],
                y=pareto_set[:, 1],
                mode="markers",
                marker={**markers, "color": "blue"},
                name="Pareto Set",
            ),
            row=1,
            col=1,
        )
        self._set_axis_limits(fig, 1, 1, pareto_set[:, 0], pareto_set[:, 1])
        fig.update_xaxes(title_text="$x_1$", row=1, col=1)
        fig.update_yaxes(title_text="$x_2$", row=1, col=1)
        self._add_description(
            fig, 1, 1, "Shows the Pareto set in the original decision variable space."
        )

        # Objective space
        fig.add_trace(
            go.Scatter(
                x=pareto_front[:, 0],
                y=pareto_front[:, 1],
                mode="markers",
                marker={**markers, "color": "green"},
                name="Pareto Front",
            ),
            row=1,
            col=2,
        )
        self._set_axis_limits(fig, 1, 2, pareto_front[:, 0], pareto_front[:, 1])
        fig.update_xaxes(title_text="$f_1$", row=1, col=2)
        fig.update_yaxes(title_text="$f_2$", row=1, col=2)
        self._add_description(
            fig,
            1,
            2,
            "Shows the Pareto front in the original objective function space.",
        )

        # Decision vs Objective
        fig.add_trace(
            go.Scatter(
                x=pareto_set[:, 0],
                y=pareto_front[:, 0],
                mode="markers",
                marker={**markers, "color": "purple"},
                name="x₁ vs f₁",
            ),
            row=1,
            col=3,
        )
        fig.add_trace(
            go.Scatter(
                x=pareto_set[:, 1],
                y=pareto_front[:, 1],
                mode="markers",
                marker={**markers, "color": "orange"},
                name="x₂ vs f₂",
            ),
            row=1,
            col=3,
        )
        combined_x = np.concatenate([pareto_set[:, 0], pareto_set[:, 1]])
        combined_y = np.concatenate([pareto_front[:, 0], pareto_front[:, 1]])
        self._set_axis_limits(fig, 1, 3, combined_x, combined_y)
        fig.update_xaxes(title_text="Decision variables ($x$)", row=1, col=3)
        fig.update_yaxes(title_text="Objective values ($f$)", row=1, col=3)
        self._add_description(
            fig,
            1,
            3,
            "Relationship between decision variables and their corresponding objectives.",
        )

    def _add_normalized_spaces(self, fig, pareto_set, pareto_front):
        """Visualize normalized decision and objective spaces."""
        scaler = MinMaxScaler()
        norm_set = scaler.fit_transform(pareto_set)
        norm_front = scaler.fit_transform(pareto_front)

        markers = dict(size=6, opacity=0.7)

        # Normalized Decision
        fig.add_trace(
            go.Scatter(
                x=norm_set[:, 0],
                y=norm_set[:, 1],
                mode="markers",
                marker={**markers, "color": "blue"},
                name="Norm Pareto Set",
            ),
            row=2,
            col=1,
        )
        self._set_axis_limits(fig, 2, 1, norm_set[:, 0], norm_set[:, 1])
        fig.update_xaxes(title_text="Norm $x_1$", row=2, col=1)
        fig.update_yaxes(title_text="Norm $x_2$", row=2, col=1)
        self._add_description(
            fig,
            2,
            1,
            "Normalized decision space showing scaled decision variables between 0 and 1.",
        )

        # Normalized Objective
        fig.add_trace(
            go.Scatter(
                x=norm_front[:, 0],
                y=norm_front[:, 1],
                mode="markers",
                marker={**markers, "color": "green"},
                name="Norm Pareto Front",
            ),
            row=2,
            col=2,
        )
        self._set_axis_limits(fig, 2, 2, norm_front[:, 0], norm_front[:, 1])
        fig.update_xaxes(title_text="Norm $f_1$", row=2, col=2)
        fig.update_yaxes(title_text="Norm $f_2$", row=2, col=2)
        self._add_description(
            fig,
            2,
            2,
            "Normalized objective space showing scaled objective values between 0 and 1.",
        )

        # Combined Normalized Decision vs Objective
        fig.add_trace(
            go.Scatter(
                x=norm_set[:, 0],
                y=norm_front[:, 0],
                mode="markers",
                marker={**markers, "color": "purple"},
                name="Norm x₁ vs f₁",
            ),
            row=2,
            col=3,
        )
        fig.add_trace(
            go.Scatter(
                x=norm_set[:, 1],
                y=norm_front[:, 1],
                mode="markers",
                marker={**markers, "color": "orange"},
                name="Norm x₂ vs f₂",
            ),
            row=2,
            col=3,
        )
        combined_x = np.concatenate([norm_set[:, 0], norm_set[:, 1]])
        combined_y = np.concatenate([norm_front[:, 0], norm_front[:, 1]])
        self._set_axis_limits(fig, 2, 3, combined_x, combined_y)
        fig.update_xaxes(title_text="Normalized $x$", row=2, col=3)
        fig.update_yaxes(title_text="Normalized $f$", row=2, col=3)
        self._add_description(
            fig,
            2,
            3,
            "Normalized relationship between decision variables and objectives.",
        )

    def _add_f1_relationships(self, fig, pareto_set, pareto_front):
        """Add f₁ relationships with f₂, x₁, x₂ using cubic splines."""
        idx = np.argsort(pareto_front[:, 0])
        f1 = pareto_front[idx, 0]
        f2, x1, x2 = pareto_front[idx, 1], pareto_set[idx, 0], pareto_set[idx, 1]

        spline_f2 = CubicSpline(f1, f2)
        spline_x1 = CubicSpline(f1, x1)
        spline_x2 = CubicSpline(f1, x2)

        f1_interp = np.linspace(f1.min(), f1.max(), 100)

        for i, (y, y_interp, label, color) in enumerate(
            [
                (f2, spline_f2(f1_interp), "$f_2$", "purple"),
                (x1, spline_x1(f1_interp), "$x_1$", "orange"),
                (x2, spline_x2(f1_interp), "$x_2$", "brown"),
            ]
        ):
            fig.add_trace(
                go.Scatter(
                    x=f1,
                    y=y,
                    mode="markers",
                    marker=dict(color=color, size=6),
                    name=f"Data Points {label}",
                ),
                row=3,
                col=i + 1,
            )
            fig.add_trace(
                go.Scatter(
                    x=f1_interp,
                    y=y_interp,
                    mode="lines",
                    line=dict(color="red"),
                    name=f"Cubic Spline {label}",
                ),
                row=3,
                col=i + 1,
            )
            self._set_axis_limits(fig, 3, i + 1, f1, np.concatenate([y, y_interp]))
            fig.update_xaxes(title_text="$f_1$", row=3, col=i + 1)
            fig.update_yaxes(title_text=label, row=3, col=i + 1)
            self._add_description(
                fig,
                3,
                i + 1,
                f"Shows the relationship between $f_1$ and {label} with cubic spline interpolation.",
            )

    def _add_parallel_coordinates(self, fig, pareto_set, pareto_front):
        """Add parallel coordinates plot for combined variables."""

        scaler = MinMaxScaler()
        norm_set = scaler.fit_transform(pareto_set)
        norm_front = scaler.fit_transform(pareto_front)

        data = np.hstack((norm_set, norm_front))
        dims = [
            dict(label=f"x{i + 1}", values=data[:, i]) for i in range(norm_set.shape[1])
        ] + [
            dict(label=f"f{i + 1}", values=data[:, norm_set.shape[1] + i])
            for i in range(norm_front.shape[1])
        ]

        fig.add_trace(
            go.Parcoords(
                line=dict(
                    color=norm_front[:, 0], colorscale="Viridis", showscale=False
                ),
                dimensions=dims,
            ),
            row=4,
            col=1,
        )
        self._add_description(
            fig,
            6,
            1,
            "Parallel coordinates showing all decision variables and objectives to visualize trade-offs.",
        )
