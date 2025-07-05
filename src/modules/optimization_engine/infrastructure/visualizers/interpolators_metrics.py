import pandas as pd
import plotly.express as px

from ...domain.analysis.interfaces.base_visualizer import (
    BaseDataVisualizer,
)


class PlotlyIntrepolatorsMetricsVisualizer(BaseDataVisualizer):
    """
    A concrete visualizer that uses Plotly Express to plot model performance.
    This is an infrastructure component.
    """

    def plot(self, data: dict[str, list[float]]) -> None:
        """
        Creates and saves an interactive Plotly box plot from a dictionary of metrics.

        Args:
            data (dict[str, list[float]]): A dictionary where keys are method names
                                           and values are lists of metric values.
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
        # This is a key step for plotting with Plotly Express
        long_data = []
        for method, metrics_list in data.items():
            for metric_value in metrics_list:
                long_data.append({"method": method, "metric_value": metric_value})

        df = pd.DataFrame(long_data)

        # Create the plot using Plotly Express
        fig = px.box(
            df,
            x="method",
            y="metric_value",
            color="method",
            points="all",
            title="Mean Squared Error by Method (Decision Mapper)",
            labels={"method": "Method", "metric_value": "Mean Squared Error (MSE)"},
            hover_name="metric_value",
        )

        # Customize layout
        fig.update_layout(
            xaxis_title="Method",
            yaxis_title="Mean Squared Error (MSE)",
            showlegend=False,
            template="plotly_white",
            yaxis_type="log",  # Use a logarithmic scale for better visualization of small values
        )

        # Display the plot
        fig.show()
