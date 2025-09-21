"""Infrastructure adapter wrapping a collection of forward estimators."""

from typing import Sequence

import numpy as np

from .....domain.assurance.decision_validation.interfaces import ForwardModel
from .....domain.modeling.interfaces.base_estimator import BaseEstimator


class ForwardEnsembleAdapter(ForwardModel):
    def __init__(self, estimators: Sequence[BaseEstimator]):
        if not estimators:
            raise ValueError("ForwardEnsembleAdapter requires at least one estimator")
        self._estimators = list(estimators)

    def _predict_all(self, X: np.ndarray) -> np.ndarray:
        preds = [est.predict(X) for est in self._estimators]
        return np.stack(preds, axis=0)

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return self._predict_all(X).mean(axis=0)


__all__ = ["ForwardEnsembleAdapter"]
