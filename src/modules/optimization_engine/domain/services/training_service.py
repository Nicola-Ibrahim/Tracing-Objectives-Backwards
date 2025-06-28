import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split

from ..interpolation.interfaces.base_inverse_decision_mappers import (
    BaseInverseDecisionMapper,
)
from ..interpolation.interfaces.base_metric import BaseValidationMetric
from ..interpolation.interfaces.base_normalizer import BaseNormalizer


class DecisionMapperTrainingService:
    """
    Domain service responsible for handling the data splitting, normalization,
    and the core training/prediction workflow for interpolators.
    It encapsulates the training and prediction concerns for better SRP adherence.
    """

    def __init__(
        self,
        validation_metric: BaseValidationMetric,
        decisions_normalizer: BaseNormalizer,
        objectives_normalizer: BaseNormalizer,
    ):
        """
        Initializes the service with a validation metric dependency.
        """
        self._validation_metric = validation_metric
        self._decisions_normalizer = decisions_normalizer
        self._objectives_normalizer = objectives_normalizer

    def train(
        self,
        inverse_decision_mapper: BaseInverseDecisionMapper,
        objectives: NDArray[np.floating],  # Objective data (input to the mapper)
        decisions: NDArray[np.floating],  # Decision data (output from the mapper)
        test_size: float = 0.33,
        random_state: int = 42,
    ) -> tuple[
        BaseInverseDecisionMapper,
        dict[str, float],
    ]:
        """
        Splits data, normalizes it, trains the interpolator, and calculates validation metrics.

        Args:
            inverse_decision_mapper: An unfitted instance of BaseInverseDecisionMapper.
            objectives: Raw input data for the mapper (Objectives).
            decisions: Raw output data for the mapper (Decisions).
            test_size: Proportion of data for validation.
            random_state: Seed for reproducibility.

        Returns:
            A tuple containing:
            - fitted_inverse_decision_mapper (BaseInverseDecisionMapper): The trained interpolator.
            - objectives_normalizer (BaseNormalizer): The fitted normalizer for objective data.
            - decisions_normalizer (BaseNormalizer): The fitted normalizer for decision data.
            - metrics (Dict): A dictionary of calculated validation metrics.
        """
        # Split data into train and validation sets
        objectives_train, objectives_val, decisions_train, decisions_val = (
            train_test_split(
                objectives, decisions, test_size=test_size, random_state=random_state
            )
        )

        # Normalize training and validation data
        objectives_train = self._objectives_normalizer.fit_transform(objectives_train)
        objectives_val = self._objectives_normalizer.transform(objectives_val)
        decisions_train = self._decisions_normalizer.fit_transform(decisions_train)

        # Fit the interpolator instance on normalized data
        inverse_decision_mapper.fit(
            objectives=objectives_train,  # Objectives are the input to the inverse mapper
            decisions=decisions_train,  # Decisions are the output of the inverse mapper
        )

        # Predict decision values on the validation set
        decisions_pred_val = inverse_decision_mapper.predict(objectives_val)

        # Inverse-transform predictions to original scale
        decisions_pred_val_2_original = self._decisions_normalizer.inverse_transform(
            decisions_pred_val
        )

        # Calculate validation metrics using the injected metric
        metrics = {
            self._validation_metric.name: self._validation_metric.calculate(
                y_true=decisions_val, y_pred=decisions_pred_val_2_original
            )
        }

        # Call the dedicated plotting method
        self._plot_validation_results(
            objectives_train=objectives_train,
            objectives_val=objectives_val,
            decisions_train=decisions_train,
            decisions_val=decisions_val,
            decisions_pred_val=decisions_pred_val_2_original,
            validation_metric_name=self._validation_metric.name,
        )

        # Return fitted instance, fitted normalizers, and metrics
        return inverse_decision_mapper, metrics

    def predict(
        self,
        fitted_inverse_decision_mapper: BaseInverseDecisionMapper,
        target_objectives_norm: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Generates predictions using a fitted interpolator and inverse-transforms them.

        Args:
            fitted_inverse_decision_mapper: A pre-fitted instance of BaseInverseDecisionMapper.
            target_objectives_norm: Normalized objective values for which to make predictions.

        Returns:
            predicted_decisions (NDArray): Predicted decision values in their original scale.
        """
        # Predict normalized values (these are normalized decision values)
        predicted_decisions_norm = fitted_inverse_decision_mapper.predict(
            target_objectives_norm
        )

        # Inverse-transform predictions to original scale
        predicted_decisions = self._decisions_normalizer.inverse_transform(
            predicted_decisions_norm
        )

        return predicted_decisions

    def _plot_validation_results(
        self,
        objectives_train: NDArray[np.floating],
        objectives_val: NDArray[np.floating],
        decisions_train: NDArray[np.floating],
        decisions_val: NDArray[np.floating],
        decisions_pred_val: NDArray[np.floating],
        validation_metric_name: str,
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
        residuals_dim2 = (
            decisions_val[:, 1] - decisions_pred_val[:, 1]
        )  # Explicitly calculate for new plot

        # --- Create a 5x2 subplot figure with specific titles ---
        # Increased rows from 4 to 5 to accommodate the new residuals plot
        fig = make_subplots(
            rows=5,  # Now 5 rows
            cols=2,
            subplot_titles=(
                "Model Performance: Actual vs. Predicted Decisions",  # (1,1)
                "Input Data Distribution (Objective Space)",  # (1,2)
                "Output Data Distribution (Decision Space)",  # (2,1)
                "Prediction Error Magnitude",  # (2,2)
                "Residuals vs. Predicted (Decision Dim 1)",  # (3,1)
                "Residuals vs. Predicted (Decision Dim 2)",  # (3,2)
                "Residuals in 2D Space (Dim 1 vs. Dim 2)",  # NEW (4,1)
                "Error Magnitude vs. Predicted Value (Overall)",  # Shifted to (4,2)
                "Distribution of Prediction Errors",  # Shifted to (5,1)
                "",  # Empty title for (5,2)
            ),
            horizontal_spacing=0.08,  # Spacing between columns
            vertical_spacing=0.1,  # Adjusted vertical spacing for better annotation placement
        )

        # --- Subplot 1 (Row 1, Col 1): Actual vs. Predicted Decisions ---
        # Retaining go.Scatter for specific marker control and connecting lines
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
        # Add lines connecting actual to predicted points to visualize error vectors
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

        # --- Subplot 2 (Row 1, Col 2): Objective Space (Merged) ---
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

        # --- Subplot 3 (Row 2, Col 1): Decision Space (Merged) ---
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

        # --- Subplot 4 (Row 2, Col 2): Error Magnitude ---
        fig_errors_overall = px.scatter(x=objectives_val[:, 0], y=errors)
        for trace in fig_errors_overall.data:
            trace.update(
                name="Error Magnitude",
                marker=dict(color="purple", size=8, opacity=0.7, symbol="cross"),
            )
            fig.add_trace(trace, row=2, col=2)

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
        fig_residuals2 = px.scatter(
            x=decisions_pred_val[:, 1], y=residuals_dim2
        )  # Using Dim 2 predicted and residuals
        for trace in fig_residuals2.data:
            trace.update(
                name="Residuals (Dim 2)",
                marker=dict(color="darkcyan", size=8, opacity=0.7, symbol="star-open"),
            )  # New color/symbol
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

        # --- NEW Subplot 7 (Row 4, Col 1): Residuals in 2D Space (Dim 1 vs. Dim 2) ---
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

        # --- Subplot 8 (Row 4, Col 2): Error Magnitude vs. Predicted Value (Overall) ---
        # Shifted from Row 4, Col 2 to a new position
        fig_err_vs_pred = px.scatter(x=decisions_pred_val[:, 0], y=errors)
        for trace in fig_err_vs_pred.data:
            trace.update(
                name="Error Mag vs. Pred Val",
                marker=dict(color="brown", size=8, opacity=0.7, symbol="diamond-open"),
            )
            fig.add_trace(trace, row=4, col=2)

        # --- Subplot 9 (Row 5, Col 1): Distribution of Prediction Errors (Histogram) ---
        # Shifted from Row 4, Col 1 to a new position
        histogram_fig = px.histogram(
            x=errors, nbins=20, color_discrete_sequence=["teal"]
        )
        for trace in histogram_fig.data:
            fig.add_trace(trace, row=5, col=1)

        # --- Update subplot axes titles for clarity ---
        fig.update_xaxes(title_text="Decision Dimension 1", row=1, col=1)
        fig.update_yaxes(title_text="Decision Dimension 2", row=1, col=1)
        fig.update_xaxes(title_text="Objective Dimension 1", row=1, col=2)
        fig.update_yaxes(title_text="Objective Dimension 2", row=1, col=2)
        fig.update_xaxes(title_text="Decision Dimension 1", row=2, col=1)
        fig.update_yaxes(title_text="Decision Dimension 2", row=2, col=1)
        fig.update_xaxes(title_text="Objective Dimension 1", row=2, col=2)
        fig.update_yaxes(
            title_text=f"Error Magnitude ({validation_metric_name.upper()})",
            row=2,
            col=2,
        )
        fig.update_xaxes(title_text="Predicted Decision Dim 1", row=3, col=1)
        fig.update_yaxes(title_text="Residual (Actual - Predicted)", row=3, col=1)
        fig.update_xaxes(title_text="Predicted Decision Dim 2", row=3, col=2)
        fig.update_yaxes(title_text="Residual (Actual - Predicted)", row=3, col=2)
        fig.update_xaxes(
            title_text="Residual Dimension 1", row=4, col=1
        )  # New axis title
        fig.update_yaxes(
            title_text="Residual Dimension 2", row=4, col=1
        )  # New axis title
        fig.update_xaxes(title_text="Predicted Value (Decision Dim 1)", row=4, col=2)
        fig.update_yaxes(
            title_text=f"Error Magnitude ({validation_metric_name.upper()})",
            row=4,
            col=2,
        )
        fig.update_xaxes(
            title_text=f"Error Magnitude ({validation_metric_name.upper()})",
            row=5,
            col=1,
        )
        fig.update_yaxes(title_text="Count", row=5, col=1)

        # --- CALCULATE DYNAMIC ANNOTATION POSITIONS FOR ALIGNMENT ---
        y_domain_row1 = fig.layout.yaxis.domain
        y_domain_row2 = fig.layout.yaxis3.domain
        y_domain_row3 = fig.layout.yaxis5.domain
        y_domain_row4 = fig.layout.yaxis7.domain
        y_domain_row5 = fig.layout.yaxis9.domain

        y_offset = 0.04
        y_pos_row1 = y_domain_row1[0] - y_offset
        y_pos_row2 = y_domain_row2[0] - y_offset
        y_pos_row3 = y_domain_row3[0] - y_offset
        y_pos_row4 = y_domain_row4[0] - y_offset
        y_pos_row5 = y_domain_row5[0] - y_offset

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
                y=y_pos_row1,
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
                y=y_pos_row1,
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
                y=y_pos_row2,
                showarrow=False,
                align="center",
                font=dict(size=10, color="grey"),
                yanchor="top",
            ),
            # Annotation for Subplot 4 (Row 2, Col 2)
            go.layout.Annotation(
                text="Plots the error magnitude vs. an input dimension.<br>"
                "A consistent scatter indicates uniform model performance across the input space.",
                xref="paper",
                yref="paper",
                x=center_x_col2,
                y=y_pos_row2,
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
                y=y_pos_row3,
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
                y=y_pos_row3,
                showarrow=False,
                align="center",
                font=dict(size=10, color="grey"),
                yanchor="top",
            ),
            # Annotation for Subplot 7 (Row 4, Col 1) - NEW
            go.layout.Annotation(
                text="Shows residuals of Decision Dim 1 vs. Dim 2.<br>"
                "Ideally, points should be clustered near (0,0).<br>"
                "Clustering in other quadrants could suggest correlated errors.",
                xref="paper",
                yref="paper",
                x=center_x_col1,
                y=y_pos_row4,
                showarrow=False,
                align="center",
                font=dict(size=10, color="grey"),
                yanchor="top",
            ),
            # Annotation for Subplot 8 (Row 4, Col 2)
            go.layout.Annotation(
                text="Plots overall error magnitude against predicted values (Decision Dim 1).<br>"
                "Helps identify if error size changes systematically with predicted value.",
                xref="paper",
                yref="paper",
                x=center_x_col2,
                y=y_pos_row4,
                showarrow=False,
                align="center",
                font=dict(size=10, color="grey"),
                yanchor="top",
            ),
            # Annotation for Subplot 9 (Row 5, Col 1)
            go.layout.Annotation(
                text="Histogram showing the frequency of different error magnitudes.<br>"
                "Ideally, errors should be centered around zero and normally distributed.",
                xref="paper",
                yref="paper",
                x=center_x_col1,
                y=y_pos_row5,
                showarrow=False,
                align="center",
                font=dict(size=10, color="grey"),
                yanchor="top",
            ),
        ]

        # --- Update final layout with overall titles, axes, and annotations ---
        fig.update_layout(
            title_text="Comprehensive Model Validation Dashboard",
            height=2400,  # Increased height significantly for the new row
            showlegend=True,
            hovermode="closest",
            annotations=annotations,
            margin=dict(b=500, t=80),
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
