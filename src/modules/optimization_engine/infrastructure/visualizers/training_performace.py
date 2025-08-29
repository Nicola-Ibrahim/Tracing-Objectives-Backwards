import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import ShuffleSplit, train_test_split

from ...application.services.cross_validation import _clone
from ...domain.analysis.interfaces.base_visualizer import BaseDataVisualizer
from ...domain.model_evaluation.interfaces.base_validation_metric import (
    BaseValidationMetric,
)
from ...domain.model_management.interfaces.base_estimator import (
    BaseEstimator,
    ProbabilisticEstimator,
)


class LearningCurveVisualizer(BaseDataVisualizer):
    """
    A visualizer for plotting learning curves. It calculates and plots
    the training and validation scores as a function of training set size.
    """

    def __init__(self, metric: BaseValidationMetric, score_name: str = "Score"):
        """
        Initializes the visualizer with a scoring metric and a name for the score.

        Args:
            metric (BaseValidationMetric): The metric used for scoring (e.g., Accuracy, MAE).
            score_name (str): The name of the score to be displayed on the plot's y-axis.
        """
        super().__init__()
        self._metric = metric
        self.score_name = score_name
        self.training_scores = []
        self.validation_scores = []
        self.train_sizes = []

    def _generate_data(
        self,
        estimator: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        train_sizes: np.ndarray,
        verbose: bool = False,
    ):
        """
        Generates the learning curve data by training on subsets of the training data
        and scoring against both the training subset and the fixed test set.
        """
        self.train_sizes = train_sizes
        self.training_scores = []
        self.validation_scores = []

        for train_size in train_sizes:
            # Use `train_test_split` to get a fixed-size subset for training
            X_subset, _, y_subset, _ = train_test_split(
                X_train,
                y_train,
                train_size=train_size,
                random_state=42,  # Use a fixed random state for reproducibility
            )

            # Clone the model to ensure a fresh start for each subset
            cloned_mapper = _clone(estimator)
            cloned_mapper.fit(X=X_subset, y=y_subset)

            # Get predictions for scoring
            if isinstance(cloned_mapper, ProbabilisticEstimator):
                # For probabilistic models, use the mean for a point-based metric
                y_pred_train = cloned_mapper.predict(X_subset, mode="mean")
                y_pred_val = cloned_mapper.predict(X_test, mode="mean")
            else:
                y_pred_train = cloned_mapper.predict(X_subset)
                y_pred_val = cloned_mapper.predict(X_test)

            # Calculate and store scores
            self.training_scores.append(
                self._metric.calculate(y_true=y_subset, y_pred=y_pred_train)
            )
            self.validation_scores.append(
                self._metric.calculate(y_true=y_test, y_pred=y_pred_val)
            )

            if verbose:
                print(
                    f"Trained on {train_size} samples. Train Score: {self.training_scores[-1]:.4f}, Test Score: {self.validation_scores[-1]:.4f}"
                )

    def plot(self, data: dict) -> None:
        """
        Generates and displays the learning curve figure.

        Args:
            data (dict): A dictionary containing all the necessary data for plotting, including:
                - estimator: The model to analyze
                - X_train: Training objectives data
                - y_train: Training decisions data
                - X_test: Test objectives data
                - y_test: Test decisions data
                - train_sizes: Array of training set sizes
        """
        self._generate_data(
            estimator=data["estimator"],
            X_train=data["X_train"],
            y_train=data["y_train"],
            X_test=data["X_test"],
            y_test=data["y_test"],
            train_sizes=data["train_sizes"],
        )

        fig = go.Figure()

        print(self.training_scores)
        print(self.validation_scores)

        # Training scores line
        fig.add_trace(
            go.Scatter(
                x=self.train_sizes,
                y=self.training_scores,
                mode="lines+markers",
                name="Training Score",
                line=dict(color="blue"),
            )
        )

        # Validation scores line
        fig.add_trace(
            go.Scatter(
                x=self.train_sizes,
                y=self.validation_scores,
                mode="lines+markers",
                name="Test Score",
                line=dict(color="orange"),
            )
        )

        fig.update_layout(
            title=f"Learning Curve ({self.score_name})",
            xaxis_title="Number of Training Samples",
            yaxis_title=self.score_name,
            legend_title="Score",
        )

        fig.show()
