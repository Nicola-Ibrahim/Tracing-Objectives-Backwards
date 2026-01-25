import numpy as np

from modules.modeling.domain.interfaces.base_estimator import BaseEstimator


class ForwardSimulator:
    """Predicts objectives for candidate decisions using a forward model."""

    def predict(
        self,
        forward_estimator: BaseEstimator,
        candidates_raw: np.ndarray,
    ) -> np.ndarray:
        """
        Returns predicted objectives (n_candidates, y_dim).
        Forward model typically expects raw decisions and returns raw objectives.
        """
        predictions = forward_estimator.predict(candidates_raw)
        predictions = np.asarray(predictions, dtype=float)

        # Normalize shapes to 2D: (n_candidates, y_dim).
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        elif predictions.ndim == 3:
            predictions = predictions.reshape(-1, predictions.shape[-1])

        return predictions
