import plotly.graph_objects as go
from analyzing.domain.services.base_plotter import BaseParetoPlotter
from plotly.subplots import make_subplots


class PlotlyParetoPlotter(BaseParetoPlotter):
    def plot(self, pareto_set, pareto_front):
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[
                "Pareto Set (Decision Space)",
                "True Pareto Front (Objective Space)",
            ],
            horizontal_spacing=0.1,
        )

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

        fig.update_layout(
            height=500,
            width=1000,
            title_text="Pareto Optimization Visualization",
            showlegend=True,
        )
        fig.show()


def plot_pareto_visualizations(pareto_set, pareto_front):
    """Simplified Pareto visualization:
    - Row 1: Pareto Set and Pareto Front
    - Row 2: Overlay of both
    """

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

    fig.update_xaxes(title_text="$x_1$", row=2, col=1)
    fig.update_yaxes(title_text="$x_2$", row=2, col=1)

    fig.update_xaxes(title_text="$f_1$", row=2, col=2)
    fig.update_yaxes(title_text="$f_2$", row=2, col=2)

    fig.show()
