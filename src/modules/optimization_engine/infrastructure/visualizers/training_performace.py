import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots

from ...domain.visualization.interfaces.base_visualizer import (
    BaseVisualizer,
)


class PlotlyTrainingPerformanceVisualizer(BaseVisualizer):
    def plot(
        self,
        objectives_train: NDArray[np.floating],
        objectives_val: NDArray[np.floating],
        decisions_train: NDArray[np.floating],
        decisions_val: NDArray[np.floating],
        decisions_pred_val: NDArray[np.floating],
    ) -> None:
        """
        Generates and displays interactive Plotly plots to visualize validation results.
        This is a dedicated helper method to separate plotting logic from training.

        Args:
            objectives_train (NDArray[np.floating]): Training set objective values.
            objectives_val (NDArray[np.floating]): Validation set objective values.
            decisions_train (NDArray[np.floating]): Training set decision values.
            decisions_val (NDArray[np.floating]): Validation set actual decision values.
            decisions_pred_val (NDArray[np.floating]): Validation set predicted decision values.
            validation_metric_name (str): Name of the validation metric (e.g., "MSE").
        """

        # --- Calculate the error magnitude for each validation point ---
        errors = np.linalg.norm(decisions_val - decisions_pred_val, axis=1)

        # Calculate residuals for each decision dimension
        residuals_dim1 = decisions_val[:, 0] - decisions_pred_val[:, 0]
        residuals_dim2 = decisions_val[:, 1] - decisions_pred_val[:, 1]

        # Check the number of objective dimensions for plotting
        num_X_dims = objectives_val.shape[1]
        num_rows = 5
        subplot_titles_list = [
            "Model Performance: Actual vs. Predicted Decisions",  # (1,1)
            "Input Data Distribution (Objective Space)",  # (1,2)
            "Output Data Distribution (Decision Space)",  # (2,1)
            "Error Magnitude vs. Predicted Value",  # (2,2)
            "Residuals vs. Predicted (Decision Dim 1)",  # (3,1)
            "Residuals vs. Predicted (Decision Dim 2)",  # (3,2)
            "Residuals in 2D Space (Dim 1 vs. Dim 2)",  # (4,1)
            "Distribution of Prediction Errors",  # (4,2)
        ]

        # Dynamically add titles for Error Magnitude vs. Objective Dimensions
        error_magnitude_titles = []
        for i in range(num_X_dims):
            error_magnitude_titles.append(
                f"Error Magnitude vs. Objective Dimension {i + 1}"
            )

        # Determine the number of rows needed for the error magnitude plots
        num_error_mag_rows = (
            num_X_dims + 1
        ) // 2  # 1 row for 1-2 dims, 2 for 3-4, etc.
        num_rows += num_error_mag_rows

        # Add the dynamically generated titles to the subplot titles list
        subplot_titles_list.extend(error_magnitude_titles)
        # Pad with empty strings if needed to maintain pairs
        if len(subplot_titles_list) % 2 != 0:
            subplot_titles_list.append("")

        # --- Create a dynamic subplot figure with specific titles ---
        fig = make_subplots(
            rows=num_rows,
            cols=2,
            subplot_titles=tuple(subplot_titles_list),
            horizontal_spacing=0.08,
            vertical_spacing=0.07,
        )

        # ===============================================================
        # --- Group 1: Prediction Performance & Data Distribution ---
        # ===============================================================

        # --- Subplot 1 (Row 1, Col 1): Actual vs. Predicted Decisions ---
        fig.add_trace(
            go.Scatter(
                x=decisions_val[:, 0],
                y=decisions_val[:, 1],
                mode="markers",
                marker=dict(
                    color="green",
                    size=10,
                    symbol="circle",
                    line=dict(width=1, color="DarkSlateGrey"),
                ),
                name="Actual (Validation Set)",
                hovertemplate="Actual Dec 1: %{x}<br>Actual Dec 2: %{y}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=decisions_pred_val[:, 0],
                y=decisions_pred_val[:, 1],
                mode="markers",
                marker=dict(
                    color="red",
                    size=12,
                    symbol="x",
                    line=dict(width=2, color="DarkRed"),
                ),
                name="Predicted (Validation Set)",
                hovertemplate="Pred Dec 1: %{x}<br>Pred Dec 2: %{y}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        for i in range(decisions_val.shape[0]):
            fig.add_trace(
                go.Scatter(
                    x=[decisions_val[i, 0], decisions_pred_val[i, 0]],
                    y=[decisions_val[i, 1], decisions_pred_val[i, 1]],
                    mode="lines",
                    line=dict(color="orange", width=1, dash="dot"),
                    opacity=0.8,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

        # --- Subplot 2 (Row 1, Col 2): Objective Space Distribution ---
        df_objectives = pd.DataFrame(
            {
                "dim1": np.concatenate([objectives_train[:, 0], objectives_val[:, 0]]),
                "dim2": np.concatenate([objectives_train[:, 1], objectives_val[:, 1]]),
                "type": ["Training Objectives"] * objectives_train.shape[0]
                + ["Validation Objectives"] * objectives_val.shape[0],
            }
        )
        fig_obj = px.scatter(
            df_objectives,
            x="dim1",
            y="dim2",
            color="type",
            color_discrete_map={
                "Training Objectives": "blue",
                "Validation Objectives": "green",
            },
            symbol="type",
            symbol_map={
                "Training Objectives": "circle",
                "Validation Objectives": "square",
            },
            opacity=0.6,
        )
        for trace in fig_obj.data:
            if trace.name == "Validation Objectives":
                trace.marker.opacity = 0.8
            fig.add_trace(trace, row=1, col=2)

        # --- Subplot 3 (Row 2, Col 1): Decision Space Distribution ---
        df_decisions = pd.DataFrame(
            {
                "dim1": np.concatenate([decisions_train[:, 0], decisions_val[:, 0]]),
                "dim2": np.concatenate([decisions_train[:, 1], decisions_val[:, 1]]),
                "type": ["Training Decisions"] * decisions_train.shape[0]
                + ["Validation Decisions"] * decisions_val.shape[0],
            }
        )
        fig_dec = px.scatter(
            df_decisions,
            x="dim1",
            y="dim2",
            color="type",
            color_discrete_map={
                "Training Decisions": "blue",
                "Validation Decisions": "green",
            },
            symbol="type",
            symbol_map={
                "Training Decisions": "circle",
                "Validation Decisions": "square",
            },
            opacity=0.6,
        )
        for trace in fig_dec.data:
            if trace.name == "Validation Decisions":
                trace.marker.opacity = 0.8
            fig.add_trace(trace, row=2, col=1)

        # --- Subplot 4 (Row 2, Col 2): Error Magnitude vs. Predicted Value (Overall) ---
        fig_err_vs_pred = px.scatter(x=decisions_pred_val[:, 0], y=errors)
        for trace in fig_err_vs_pred.data:
            trace.update(
                name="Error Mag vs. Pred Val",
                marker=dict(color="brown", size=8, opacity=0.7, symbol="diamond-open"),
            )
            fig.add_trace(trace, row=2, col=2)

        # ===============================================================
        # --- Group 2: Residuals Analysis ---
        # ===============================================================

        # --- Subplot 5 (Row 3, Col 1): Residuals vs. Predicted (Decision Dim 1) ---
        fig_residuals1 = px.scatter(x=decisions_pred_val[:, 0], y=residuals_dim1)
        for trace in fig_residuals1.data:
            trace.update(
                name="Residuals (Dim 1)",
                marker=dict(
                    color="darkorange", size=8, opacity=0.7, symbol="circle-open"
                ),
            )
            fig.add_trace(trace, row=3, col=1)

        fig.add_hline(
            y=0,
            line_dash="dot",
            line_color="grey",
            row=3,
            col=1,
            annotation_text="Zero Residuals",
            annotation_position="bottom right",
        )

        # --- Subplot 6 (Row 3, Col 2): Residuals vs. Predicted (Decision Dim 2) ---
        fig_residuals2 = px.scatter(x=decisions_pred_val[:, 1], y=residuals_dim2)
        for trace in fig_residuals2.data:
            trace.update(
                name="Residuals (Dim 2)",
                marker=dict(color="darkcyan", size=8, opacity=0.7, symbol="star-open"),
            )
            fig.add_trace(trace, row=3, col=2)

        fig.add_hline(
            y=0,
            line_dash="dot",
            line_color="grey",
            row=3,
            col=2,
            annotation_text="Zero Residuals",
            annotation_position="bottom right",
        )

        # --- Subplot 7 (Row 4, Col 1): Residuals in 2D Space (Dim 1 vs. Dim 2) ---
        fig.add_trace(
            go.Scatter(
                x=residuals_dim1,
                y=residuals_dim2,
                mode="markers",
                name="2D Residuals",
                marker=dict(
                    color="red",
                    size=8,
                    opacity=0.7,
                    symbol="square-open",
                    line=dict(width=1, color="DarkRed"),
                ),
                hovertemplate="Residual Dim 1: %{x}<br>Residual Dim 2: %{y}<extra></extra>",
            ),
            row=4,
            col=1,
        )
        fig.add_vline(x=0, line_dash="dot", line_color="grey", row=4, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="grey", row=4, col=1)

        # --- Subplot 8 (Row 4, Col 2): Distribution of Prediction Errors (Histogram) ---
        histogram_fig = px.histogram(
            x=errors, nbins=20, color_discrete_sequence=["teal"]
        )
        for trace in histogram_fig.data:
            fig.add_trace(trace, row=4, col=2)

        # ===============================================================
        # --- Group 3: Error Magnitude vs. Objective Dimensions ---
        # ===============================================================

        # Dynamically add scatter plots for each objective dimension
        current_row = 5
        for i in range(num_X_dims):
            col = (i % 2) + 1  # 1, 2, 1, 2...

            fig.add_trace(
                go.Scatter(
                    x=objectives_val[:, i],
                    y=errors,
                    mode="markers",
                    name=f"Error vs. Obj Dim {i+1}",
                    marker=dict(
                        color=px.colors.qualitative.Plotly[i],
                        size=8,
                        opacity=0.7,
                        symbol="circle",
                    ),
                ),
                row=current_row,
                col=col,
            )

            # If we're on the second column, increment the row
            if col == 2:
                current_row += 1

        # --- Update subplot axes titles for clarity ---
        fig.update_xaxes(title_text="Decision Dimension 1", row=1, col=1)
        fig.update_yaxes(title_text="Decision Dimension 2", row=1, col=1)
        fig.update_xaxes(title_text="Objective Dimension 1", row=1, col=2)
        fig.update_yaxes(title_text="Objective Dimension 2", row=1, col=2)

        fig.update_xaxes(title_text="Decision Dimension 1", row=2, col=1)
        fig.update_yaxes(title_text="Decision Dimension 2", row=2, col=1)
        fig.update_xaxes(title_text="Predicted Value (Decision Dim 1)", row=2, col=2)
        fig.update_yaxes(
            title_text="Error Magnitude (MSE)",
            row=2,
            col=2,
        )

        fig.update_xaxes(title_text="Predicted Decision Dim 1", row=3, col=1)
        fig.update_yaxes(title_text="Residual (Actual - Predicted)", row=3, col=1)
        fig.update_xaxes(title_text="Predicted Decision Dim 2", row=3, col=2)
        fig.update_yaxes(title_text="Residual (Actual - Predicted)", row=3, col=2)

        fig.update_xaxes(title_text="Residual Dimension 1", row=4, col=1)
        fig.update_yaxes(title_text="Residual Dimension 2", row=4, col=1)
        fig.update_xaxes(
            title_text="Error Magnitude (MSE)",
            row=4,
            col=2,
        )
        fig.update_yaxes(title_text="Count", row=4, col=2)

        # Dynamically update axes titles for the new error magnitude plots
        for i in range(num_X_dims):
            current_row = 5 + i // 2
            col = (i % 2) + 1
            fig.update_xaxes(
                title_text=f"Objective Dimension {i+1}", row=current_row, col=col
            )
            fig.update_yaxes(
                title_text="Error Magnitude (MSE)",
                row=current_row,
                col=col,
            )

        y_domains = [fig.layout[f"yaxis{2*i+1}"].domain for i in range(num_rows)]
        y_offset = 0.03
        y_positions = [dom[0] - y_offset for dom in y_domains]

        center_x_col1 = (fig.layout.xaxis.domain[0] + fig.layout.xaxis.domain[1]) / 2
        center_x_col2 = (fig.layout.xaxis2.domain[0] + fig.layout.xaxis2.domain[1]) / 2

        annotations = [
            # Annotation for Subplot 1 (Row 1, Col 1)
            go.layout.Annotation(
                text="Visualizes predictions in the output space.<br>"
                "<span style='color:green;'>Green circles</span> are actual; <span style='color:red;'>red crosses</span> are predicted.<br>"
                "Shorter lines indicate a better fit.",
                xref="paper",
                yref="paper",
                x=center_x_col1,
                y=y_positions[0],
                showarrow=False,
                align="center",
                font=dict(size=10, color="grey"),
                yanchor="top",
            ),
            # Annotation for Subplot 2 (Row 1, Col 2)
            go.layout.Annotation(
                text="Shows the distribution of training (blue) and validation (green) sets in the input space.<br>"
                "Ensures the validation set is representative.",
                xref="paper",
                yref="paper",
                x=center_x_col2,
                y=y_positions[0],
                showarrow=False,
                align="center",
                font=dict(size=10, color="grey"),
                yanchor="top",
            ),
            # Annotation for Subplot 3 (Row 2, Col 1)
            go.layout.Annotation(
                text="Shows the distribution of the output (decision) values for both training (blue) and validation (green) sets.<br>"
                "Helps to understand the range and density of the output data.",
                xref="paper",
                yref="paper",
                x=center_x_col1,
                y=y_positions[1],
                showarrow=False,
                align="center",
                font=dict(size=10, color="grey"),
                yanchor="top",
            ),
            # Annotation for Subplot 4 (Row 2, Col 2)
            go.layout.Annotation(
                text="Plots overall error magnitude against predicted values (Decision Dim 1).<br>"
                "Helps identify if error size changes systematically with predicted value.",
                xref="paper",
                yref="paper",
                x=center_x_col2,
                y=y_positions[1],
                showarrow=False,
                align="center",
                font=dict(size=10, color="grey"),
                yanchor="top",
            ),
            # Annotation for Subplot 5 (Row 3, Col 1)
            go.layout.Annotation(
                text="Shows the difference between actual and predicted values for Decision Dim 1 vs. predicted values.<br>"
                "Random scatter around zero suggests a good fit; patterns indicate bias or issues.",
                xref="paper",
                yref="paper",
                x=center_x_col1,
                y=y_positions[2],
                showarrow=False,
                align="center",
                font=dict(size=10, color="grey"),
                yanchor="top",
            ),
            # Annotation for Subplot 6 (Row 3, Col 2)
            go.layout.Annotation(
                text="Shows the difference between actual and predicted values for Decision Dim 2 vs. predicted values.<br>"
                "Random scatter around zero suggests a good fit; patterns indicate bias or issues.",
                xref="paper",
                yref="paper",
                x=center_x_col2,
                y=y_positions[2],
                showarrow=False,
                align="center",
                font=dict(size=10, color="grey"),
                yanchor="top",
            ),
            # Annotation for Subplot 7 (Row 4, Col 1)
            go.layout.Annotation(
                text="Shows residuals of Decision Dim 1 vs. Dim 2.<br>"
                "Ideally, points should be clustered near (0,0).<br>"
                "Clustering in other quadrants could suggest correlated errors.",
                xref="paper",
                yref="paper",
                x=center_x_col1,
                y=y_positions[3],
                showarrow=False,
                align="center",
                font=dict(size=10, color="grey"),
                yanchor="top",
            ),
            # Annotation for Subplot 8 (Row 4, Col 2)
            go.layout.Annotation(
                text="Histogram showing the frequency of different error magnitudes.<br>"
                "Ideally, errors should be centered around zero and normally distributed.",
                xref="paper",
                yref="paper",
                x=center_x_col2,
                y=y_positions[3],
                showarrow=False,
                align="center",
                font=dict(size=10, color="grey"),
                yanchor="top",
            ),
        ]

        # Dynamically add annotations for the new error magnitude plots
        for i in range(num_X_dims):
            current_row_index = 4 + i // 2
            col = (i % 2) + 1
            x_pos = center_x_col1 if col == 1 else center_x_col2

            annotations.append(
                go.layout.Annotation(
                    text=f"Plots the error magnitude against objective dimension {i+1}.<br>"
                    "A consistent scatter indicates uniform model performance.",
                    xref="paper",
                    yref="paper",
                    x=x_pos,
                    y=y_positions[current_row_index],
                    showarrow=False,
                    align="center",
                    font=dict(size=10, color="grey"),
                    yanchor="top",
                )
            )

        # --- Update final layout with overall titles, axes, and annotations ---
        fig.update_layout(
            title_text="Comprehensive Model Validation Dashboard",
            height=3000,  # Increased height for the new plots
            showlegend=True,
            hovermode="closest",
            annotations=annotations,
            margin=dict(b=450, t=80),
            legend=dict(
                x=1.02,
                y=1,
                xref="paper",
                yref="paper",
                traceorder="normal",
                font=dict(family="sans-serif", size=10, color="black"),
                bgcolor="LightSteelBlue",
                bordercolor="Black",
                borderwidth=1,
            ),
        )

        fig.show()
