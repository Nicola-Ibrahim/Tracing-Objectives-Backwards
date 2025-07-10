import numpy as np
from numpy.typing import NDArray
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    Kernel,
    Matern,
)

from ....domain.interpolation.interfaces.base_inverse_decision_mapper import (
    BaseInverseDecisionMapper,
)


class GaussianProcessInverseDecisionMapper(BaseInverseDecisionMapper):
    """
    Inverse Decision Mapper using Scikit-learn's GaussianProcessRegressor.
    This model is capable of both interpolation and extrapolation, and can provide
    uncertainty estimates.
    """

    _gpr_model: GaussianProcessRegressor | None = None

    def __init__(
        self,
        kernel: Kernel | str = "Matern",
        alpha: float = 1e-10,
        n_restarts_optimizer: int = 10,
        random_state: int = 42,
    ) -> None:
        """
        Initializes the Gaussian Process Regressor mapper.

        Args:
            kernel (Kernel | str): The kernel to use for the GPR. Can be a string
                                   ('RBF' or 'Matern') or a scikit-learn kernel object.
                                   Matern is often a good default choice for real-world data.
            alpha (float): Value added to the diagonal of the kernel matrix for numerical stability.
            n_restarts_optimizer (int): Number of restarts of the optimizer for the kernel's hyperparameters.
            random_state (int): Seed for reproducibility.
        """
        super().__init__()
        self._gpr_model = None
        self._alpha = alpha
        self._n_restarts_optimizer = n_restarts_optimizer
        self._random_state = random_state

        # Handle string input for common kernels
        if isinstance(kernel, str):
            kernel_name = kernel.lower()
            if kernel_name == "rbf":
                self.kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            elif kernel_name == "matern":
                # A good default Matern kernel with a ConstantKernel for scaling
                self.kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
            else:
                raise ValueError(
                    f"Invalid kernel string '{kernel}'. Must be 'RBF' or 'Matern'."
                )
        elif isinstance(kernel, Kernel):
            self.kernel = kernel
        else:
            raise TypeError(
                "Kernel must be a string ('RBF', 'Matern') or a scikit-learn Kernel object."
            )

    def fit(
        self,
        objectives: NDArray[np.float64],
        decisions: NDArray[np.float64],
    ) -> None:
        """
        Fits the Gaussian Process Regressor model to the provided data.

        Args:
            objectives (NDArray[np.float64]): Training data in the input space (X).
            decisions (NDArray[np.float64]): Training data in the output space (y).
        """
        # 1. Call the parent's fit method for universal data validation
        super().fit(objectives, decisions)

        # 2. Instantiate and fit the GPR model
        # The GPR handles multi-output regression by fitting a separate model for each output dimension
        # when `decisions` has more than one column.
        self._gpr_model = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=self._alpha,
            n_restarts_optimizer=self._n_restarts_optimizer,
            random_state=self._random_state,
        )

        self._gpr_model.fit(objectives, decisions)

    def predict(
        self,
        target_objectives: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Predicts corresponding 'dependent' values using the fitted GPR model.

        Args:
            target_objectives (NDArray[np.float64]): The points for which to predict values.

        Returns:
            NDArray[np.float64]: Predicted values.
        """
        # 1. Perform validation specific to this method
        if self._gpr_model is None:
            raise RuntimeError("Mapper has not been fitted yet. Call fit() first.")

        if target_objectives.ndim == 1:
            target_objectives = target_objectives.reshape(-1, 1)

        if target_objectives.shape[1] != self._objective_dim:
            raise ValueError(
                f"Target objectives must have {self._objective_dim} dimensions, "
                f"but got {target_objectives.shape[1]} dimensions."
            )

        # 2. Call the fitted GPR's predict method
        # By default, GPR predicts the mean. It can also return the standard deviation
        # if `return_std=True` is passed.
        return self._gpr_model.predict(target_objectives)
