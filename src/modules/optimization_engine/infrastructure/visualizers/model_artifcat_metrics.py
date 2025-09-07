import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...domain.visualization.interfaces.base_visualizer import (
    BaseVisualizer,
)


class PlotlyModelArtifactMetricsVisualizer(BaseVisualizer):
    """
    A concrete visualizer that uses Plotly Express to plot model performance.
    This is an infrastructure component.
    """

    def plot(self, data: dict[str, list[float]]) -> None:
        """
        Creates and displays box, violin, and bar plots in a single organized layout using Plotly subplots.

        Args:
            data (dict[str, list[float]]): Keys are method names, values are lists of metric values.
        """

        # Type check the input data
        if not isinstance(data, dict):
            print(
                f"Visualizer: Expected a dictionary, but received {type(data)}. Cannot plot."
            )
            return

        if not data:
            print("Visualizer: Input dictionary is empty. Cannot plot.")
            return

        # Transform the dictionary into a long-form DataFrame suitable for Plotly
        long_data = []
        for method, metrics_list in data.items():
            for metric_value in metrics_list:
                long_data.append({"method": method, "metric_value": metric_value})

        df = pd.DataFrame(long_data)

        # Prepare mean data for bar plot
        mean_df = df.groupby("method", as_index=False)["metric_value"].mean()

        # Create subplots: 1 row, 3 columns
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=("Box Plot", "Violin Plot", "Bar Plot (Mean)"),
            shared_xaxes=False,
        )

        # Box plot
        for method in df["method"].unique():
            method_data = df[df["method"] == method]["metric_value"]
            fig.add_trace(
                go.Box(
                    y=method_data,
                    name=method,
                    boxpoints="all",
                    marker_color=None,
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        # Violin plot
        for method in df["method"].unique():
            method_data = df[df["method"] == method]["metric_value"]
            fig.add_trace(
                go.Violin(
                    y=method_data,
                    name=method,
                    box_visible=True,
                    points="all",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

        # Bar plot (mean)
        fig.add_trace(
            go.Bar(
                x=mean_df["method"],
                y=mean_df["metric_value"],
                marker_color=None,
                showlegend=False,
            ),
            row=1,
            col=3,
        )

        # Update layout
        fig.update_layout(
            title_text="Metric Comparison by Method",
            template="plotly_white",
            height=500,
            width=1200,
        )
        fig.update_xaxes(title_text="Method", row=1, col=1)
        fig.update_xaxes(title_text="Method", row=1, col=2)
        fig.update_xaxes(title_text="Method", row=1, col=3)
        fig.update_yaxes(
            title_text="Metric Value (log scale)", type="log", row=1, col=1
        )
        fig.update_yaxes(
            title_text="Metric Value (log scale)", type="log", row=1, col=2
        )
        fig.update_yaxes(title_text="Mean Metric Value", type="linear", row=1, col=3)

        # Display the plot
        fig.show()
