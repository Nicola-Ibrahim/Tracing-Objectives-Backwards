import numpy as np
from numpy.typing import NDArray
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

from ....domain.model_management.interfaces.base_inverse_decision_mapper import (
    DeterministicInverseDecisionMapper,
)


class SVRInverseDecisionMapper(DeterministicInverseDecisionMapper):
    """
    Inverse Decision Mapper using Scikit-learn's Support Vector Regressor (SVR)
    with a MultiOutputRegressor wrapper to handle multi-dimensional y.
    """

    # Now we store a single wrapped model
    _svr_model: MultiOutputRegressor | None = None

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
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> None:
        # 1. Call the parent's fit method for universal validation
        super().fit(X, y)

        # 2. Create a single SVR model to be wrapped
        base_svr = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)

        # 3. Wrap the single SVR model in a MultiOutputRegressor
        # This allows it to handle the multi-dimensional `y` output.
        self._svr_model = MultiOutputRegressor(estimator=base_svr)

        # 4. Fit the single wrapped model on the X and multi-dimensional y
        self._svr_model.fit(X, y)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        # Perform validation specific to this method
        if self._svr_model is None:
            raise RuntimeError("Mapper has not been fitted yet. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self._objective_dim:
            raise ValueError(
                f"Target X must have {self._objective_dim} dimensions, "
                f"but got {X.shape[1]} dimensions."
            )

        # Call the predict method on the single wrapped model
        return self._svr_model.predict(X)
