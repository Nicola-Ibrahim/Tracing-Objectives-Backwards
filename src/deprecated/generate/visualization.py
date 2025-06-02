import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_pareto_visualizations(
    rnd_solution_set,
    rnd_solution_objectives,
    pareto_set,
    pareto_front,
    f1_optimal,
    f2_optimal,
):
    """Pareto visualization with:
    - Row 1: Random decision & objective samples
    - Row 2: Pareto set and Pareto front
    - Row 3: Overlay of objectives and Pareto front
    """

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[
            "Sampled Decision Vectors",
            "Sampled Objective Vectors",
            "Pareto Set (Decision Space)",
            "True Pareto Front (Objective Space)",
            "Overlay: Objectives & Pareto Front",
            "",  # empty space
        ],
        horizontal_spacing=0.15,
        vertical_spacing=0.15,
    )

    # Row 1, Col 1: Sampled decisions
    fig.add_trace(
        go.Scatter(
            x=rnd_solution_set[:, 0],
            y=rnd_solution_set[:, 1],
            mode="markers",
            name="Sampled Decision Vectors",
            marker=dict(color="blue", size=5, opacity=0.5),
        ),
        row=1,
        col=1,
    )
    # Add f1_optimal and f2_optimal to sampled decision vectors subplot
    fig.add_trace(
        go.Scatter(
            x=[f1_optimal[0]],
            y=[f1_optimal[1]],
            mode="markers",
            name="f1 Optimal",
            marker=dict(color="orange", size=10, symbol="star"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[f2_optimal[0]],
            y=[f2_optimal[1]],
            mode="markers",
            name="f2 Optimal",
            marker=dict(color="purple", size=10, symbol="star"),
        ),
        row=1,
        col=1,
    )

    # Row 1, Col 2: Sampled objectives
    fig.add_trace(
        go.Scatter(
            x=rnd_solution_objectives[:, 0],
            y=rnd_solution_objectives[:, 1],
            mode="markers",
            name="Sampled Objectives",
            marker=dict(color="green", size=5, opacity=0.6),
        ),
        row=1,
        col=2,
    )

    # Row 2, Col 1: Pareto set (decision space)
    fig.add_trace(
        go.Scatter(
            x=pareto_set[:, 0],
            y=pareto_set[:, 1],
            mode="lines+markers",
            name="Pareto Set",
            marker=dict(color="red", size=6),
        ),
        row=2,
        col=1,
    )
    # Add f1_optimal and f2_optimal to Pareto set subplot
    fig.add_trace(
        go.Scatter(
            x=[f1_optimal[0]],
            y=[f1_optimal[1]],
            mode="markers",
            name="f1 Optimal",
            marker=dict(color="orange", size=10, symbol="star"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[f2_optimal[0]],
            y=[f2_optimal[1]],
            mode="markers",
            name="f2 Optimal",
            marker=dict(color="purple", size=10, symbol="star"),
        ),
        row=2,
        col=1,
    )

    # Row 2, Col 2: True Pareto front (objective space)
    fig.add_trace(
        go.Scatter(
            x=pareto_front[:, 0],
            y=pareto_front[:, 1],
            mode="lines+markers",
            name="True Pareto Front",
            marker=dict(color="red", size=6),
        ),
        row=2,
        col=2,
    )

    # Row 3, Col 1 and Col 2 combined: Overlay sampled objectives + Pareto front
    fig.add_trace(
        go.Scatter(
            x=rnd_solution_objectives[:, 0],
            y=rnd_solution_objectives[:, 1],
            mode="markers",
            name="Sampled Objectives (Overlay)",
            marker=dict(color="green", size=5, opacity=0.4),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=pareto_front[:, 0],
            y=pareto_front[:, 1],
            mode="lines+markers",
            name="Pareto Front (Overlay)",
            marker=dict(color="red", size=6),
        ),
        row=3,
        col=1,
    )

    # Set overlay subplot to span two columns by adjusting xaxis domain
    fig.update_layout(
        height=900,
        width=900,
        title_text="Pareto Optimization Visualization - Reorganized Layout",
        showlegend=True,
    )
    fig.layout["xaxis5"]["domain"] = [0, 1]  # overlay x-axis spans full width

    # Axis labels

    # Row 1
    fig.update_xaxes(title_text="$x_1$", row=1, col=1)
    fig.update_yaxes(title_text="$x_2$", row=1, col=1)

    fig.update_xaxes(title_text="$f_1$", row=1, col=2)
    fig.update_yaxes(title_text="$f_2$", row=1, col=2)

    # Row 2
    fig.update_xaxes(title_text="$x_1$", row=2, col=1)
    fig.update_yaxes(title_text="$x_2$", row=2, col=1)

    fig.update_xaxes(title_text="$f_1$", row=2, col=2)
    fig.update_yaxes(title_text="$f_2$", row=2, col=2)

    # Row 3 (overlay)
    fig.update_xaxes(title_text="$f_1$", row=3, col=1)
    fig.update_yaxes(title_text="$f_2$", row=3, col=1)

    fig.show()
