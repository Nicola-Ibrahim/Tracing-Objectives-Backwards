import numpy as np
from numpy.typing import NDArray
from sklearn.svm import SVR

from ....domain.interpolation.interfaces.base_inverse_decision_mappers import (
    BaseInverseDecisionMapper,
)


class SVRInverseDecisionMapper(BaseInverseDecisionMapper):
    """
    Inverse Decision Mapper using Scikit-learn's Support Vector Regressor (SVR).
    """

    _svr_models: list[SVR] | None = None

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        epsilon: float = 0.1,
    ) -> None:
        """
        Initializes the SVR mapper.
        Args:
            kernel (str): The kernel type ('rbf', 'linear', 'poly', etc.).
            C (float): Regularization parameter.
            epsilon (float): Epsilon-tube within which no penalty is given.
        """
        super().__init__()
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon

    def fit(
        self,
        objectives: NDArray[np.float64],
        decisions: NDArray[np.float64],
    ) -> None:
        # 1. Call the parent's fit method for universal validation
        super().fit(objectives, decisions)

        # 2. Fit a separate SVR model for each output dimension
        self._svr_models = []
        for i in range(self._decision_dim):
            model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)
            # SVR expects a 1D array for the target (decisions[:, i])
            model.fit(objectives, decisions[:, i])
            self._svr_models.append(model)

    def predict(
        self,
        target_objectives: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        # Perform validation specific to this method
        if self._svr_models is None:
            raise RuntimeError("Mapper has not been fitted yet. Call fit() first.")

        if target_objectives.ndim == 1:
            target_objectives = target_objectives.reshape(-1, 1)

        if target_objectives.shape[1] != self._objective_dim:
            raise ValueError(
                f"Target objectives must have {self._objective_dim} dimensions, "
                f"but got {target_objectives.shape[1]} dimensions."
            )

        # Call each fitted SVR model and stack the results
        predictions = []
        for model in self._svr_models:
            pred_values = model.predict(target_objectives)
            predictions.append(pred_values)

        return np.column_stack(predictions)
