import numpy as np
import pykrige.ok as krige
from numpy.typing import NDArray

from ....domain.model_management.interfaces.base_ml_mapper import (
    DeterministicMlMapper,
)


class KrigingMlMapper(DeterministicMlMapper):
    """
    Inverse Decision Mapper using PyKrige's OrdinaryKriging for 2D objective spaces.

    NOTE: This implementation is limited to objective spaces with 2 dimensions.
    """

    # We must store a list of models because the underlying library is single-output.
    _kriging_models: list[krige.OrdinaryKriging] | None = None

    def __init__(
        self,
        variogram_model: str = "linear",
        n_neighbors: int | None = 12,
    ) -> None:
        """
        Initializes the Kriging mapper.
        Args:
            variogram_model (str): The variogram model ('linear', 'gaussian', etc.).
            n_neighbors (int | None): Number of nearest neighbors to consider.
        """
        super().__init__()
        self.variogram_model = variogram_model
        self.n_neighbors = n_neighbors

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        # 1. Call the parent's fit method for universal validation
        super().fit(X, y)

        # 2. Perform specific validation
        if self._objective_dim != 2:
            raise ValueError(
                "KrigingMlMapper requires objectives with exactly 2 dimensions (x, y)."
            )

        # 3. Fit a separate Kriging model for each output dimension.
        # This is necessary because pykrige.OrdinaryKriging can only handle one output (z) at a time.
        self._kriging_models = []
        for i in range(self._decision_dim):
            model = krige.OrdinaryKriging(
                x=X[:, 0],
                y=X[:, 1],
                z=y[:, i],  # Pass only one target dimension as the output
                variogram_model=self.variogram_model,
                coordinates_type="euclidean",
            )
            self._kriging_models.append(model)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        # Perform validation specific to this method
        if self._kriging_models is None:
            raise RuntimeError("Mapper has not been fitted yet. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self._objective_dim:
            raise ValueError(
                f"Target objectives must have {self._objective_dim} dimensions, "
                f"but got {X.shape[1]} dimensions."
            )

        # Call each fitted Kriging model and stack the results
        predictions = []
        for model in self._kriging_models:
            pred_values, _ = model.execute("points", X[:, 0], X[:, 1])
            predictions.append(pred_values)

        return np.column_stack(predictions)
