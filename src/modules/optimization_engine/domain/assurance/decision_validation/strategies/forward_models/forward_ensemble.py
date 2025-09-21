from __future__ import annotations

from typing import Sequence

import numpy as np

from .....modeling.interfaces.base_estimator import BaseEstimator


class ForwardEnsemble:
    def __init__(self, estimators: Sequence[BaseEstimator]):
        if not estimators:
            raise ValueError("ForwardEnsemble requires at least one estimator")
        self._estimators = list(estimators)

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        predictions = [est.predict(X) for est in self._estimators]
        return np.stack(predictions, axis=0)

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        return self.predict_all(X).mean(axis=0)


__all__ = ["ForwardEnsemble"]
