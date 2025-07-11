# src/domain/interpolation/interfaces/base_loss_calculator.py
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.distributions import (
    MixtureSameFamily,  # Still needed for NLL specific prediction type
)

# ArrayLike for common types that can be converted
ArrayLike = npt.NDArray[np.float64] | Tensor


def to_numpy(arr: ArrayLike) -> npt.NDArray[np.float64]:
    """
    Converts a torch.Tensor to a numpy.ndarray if necessary.
    Detaches from computation graph and moves to CPU.
    """
    if isinstance(arr, Tensor):
        return arr.detach().cpu().numpy()
    return arr


def to_tensor(
    arr: ArrayLike,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> Tensor:
    """
    Converts a numpy.ndarray to a torch.Tensor if necessary.
    Allows specifying the device.
    """
    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr).to(dtype, device=device)
    # If it's already a Tensor, ensure it's on the correct device if specified
    if isinstance(arr, Tensor) and device is not None and arr.device != device:
        return arr.to(device)
    return arr


class BaseLossCalculator(ABC):
    """
    Abstract base class for all loss calculators.
    Defines the common interface for calculating loss, allowing for varied
    prediction types (including PyTorch Tensors and distributions for direct
    gradient computation).
    """

    @abstractmethod
    def calculate(
        self,
        # 'predictions' can be an ArrayLike (np.ndarray/Tensor), a dict of Tensors (for ELBO),
        # or a PyTorch distribution object (for NLL).
        predictions: ArrayLike | dict[str, Tensor | float] | MixtureSameFamily,
        targets: ArrayLike,  # Targets can be np.ndarray or Tensor
    ) -> Tensor:  # Return type is now Tensor for gradient flow
        """
        Calculate scalar loss value between predictions and targets.

        Args:
            predictions: The predicted values. Type varies per specific loss calculator.
                         Expected to be PyTorch Tensors or PyTorch-compatible objects
                         when used in a PyTorch training loop.
            targets: The true target values (NumPy array or PyTorch Tensor).
                     Will be converted to PyTorch Tensor internally if needed.

        Returns:
            torch.Tensor: The computed scalar loss value (a 0-dimensional tensor)
                          to allow for backpropagation.
        """
        pass
