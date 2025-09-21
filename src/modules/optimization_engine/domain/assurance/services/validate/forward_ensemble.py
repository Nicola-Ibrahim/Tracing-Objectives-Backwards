from typing import Sequence, list

import numpy as np

from ...domain.modeling.interfaces.base_estimator import BaseEstimator


class ForwardEnsemble:
    """
    Thin wrapper around a list of BaseEstimator forward models.

    We only need mean predictions for conformal calibration and online checks.
    """

    def __init__(self, estimators: Sequence[BaseEstimator]):
        if not estimators:
            raise ValueError("ForwardEnsemble requires at least one estimator.")
        self._estimators: list[BaseEstimator] = list(estimators)

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        """
        Predict with all ensemble members.

        Parameters
        ----------
        X : np.ndarray, shape (N, d_x)

        Returns
        -------
        np.ndarray, shape (M, N, d_y)
            Predictions from M ensemble members.
        """
        X = np.atleast_2d(X)
        preds = [est.predict(X) for est in self._estimators]
        return np.stack(preds, axis=0)

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        """
        Ensemble mean prediction.

        Returns
        -------
        np.ndarray, shape (N, d_y)
        """
        return self.predict_all(X).mean(axis=0)
