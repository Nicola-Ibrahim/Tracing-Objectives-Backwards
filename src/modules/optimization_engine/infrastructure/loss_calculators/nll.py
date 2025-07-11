import numpy as np
import numpy.typing as npt
import torch
from torch.distributions import MixtureSameFamily

from ...domain.interpolation.interfaces.base_loss_calculator import BaseLossCalculator


class NegativeLogLikelihoodLossCalculator(BaseLossCalculator):
    """
    Computes Negative Log Likelihood from MixtureSameFamily.
    Input targets are NumPy arrays. Predictions is the Mixture.
    """

    def __init__(self, mixture: MixtureSameFamily):
        if not isinstance(mixture, MixtureSameFamily):
            raise TypeError("Expected a MixtureSameFamily distribution.")
        self.mixture = mixture

    def calculate(
        self,
        predictions: npt.NDArray[np.float64],
        targets: npt.NDArray[np.float64],
    ) -> float:
        dtype = self.mixture.component_distribution.mean.dtype
        targets_tensor = torch.from_numpy(targets).to(dtype=dtype)

        loss = -self.mixture.log_prob(targets_tensor).mean()
        # return float(loss)
        return float(loss.item())
