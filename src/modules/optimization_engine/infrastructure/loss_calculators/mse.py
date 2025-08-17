import numpy as np

# Adjust import path and use the new Base
from ...domain.model_management.interfaces.base_loss_calculator import (
    ArrayLike,
    BaseLossCalculator,
    to_numpy,
)


class MSELossCalculator(BaseLossCalculator):
    """
    Calculates the Mean Squared Error (MSE) loss, operating purely on NumPy arrays.
    """

    def calculate(self, predictions: ArrayLike, targets: ArrayLike) -> float:
        """
        Calculates the MSE loss using NumPy arrays.

        Args:
            predictions: Predicted values (np.ndarray).
            targets: True target values (np.ndarray).

        Returns:
            float: The computed MSE loss.
        """
        # Ensure inputs are NumPy arrays (even if ArrayLike was used in abstract base,
        # this concrete implementation expects np.ndarray due to Base)
        preds_np = to_numpy(
            predictions
        )  # Just in case it's called with a Tensor despite type hint
        targs_np = to_numpy(targets)  # This adds a layer of safety

        return float(np.mean((preds_np - targs_np) ** 2))
