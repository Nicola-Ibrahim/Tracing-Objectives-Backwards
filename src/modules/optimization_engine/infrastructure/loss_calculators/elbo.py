from typing import Any, Dict, Union

import torch
import torch.nn.functional as F  # Import F for mse_loss
from torch import Tensor

# Adjust import path
from ...domain.model_management.interfaces.base_loss_calculator import (
    ArrayLike,  # Still used for targets
    BaseLossCalculator,
    to_tensor,  # Used for targets if they are numpy
)

# No longer need a custom ELBOPredictionsDict type as the signature will be more explicit
# based on CVAE outputs.


class ELBOLossCalculator(BaseLossCalculator):
    """
    Calculates the Evidence Lower Bound (ELBO) loss,
    combining a reconstruction loss (MSE) and a KL divergence term.
    Designed to work directly with PyTorch Tensors for gradient flow,
    taking raw CVAE outputs as predictions.
    """

    def __init__(self, recon_weight: float = 1.0, kl_weight: float = 0.1):
        """
        Initializes the ELBOLossCalculator with weights for reconstruction and KL terms.

        Args:
            recon_weight: Weight for the reconstruction loss.
            kl_weight: Weight for the KL divergence loss.
        """
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight

    # The calculate method now takes the raw components needed for ELBO
    def calculate(
        self,
        predictions: Dict[str, Tensor],  # Expects a dict with recon_x, mu, logvar
        targets: ArrayLike,  # Original input X
    ) -> Tensor:
        """
        Calculates the ELBO loss using PyTorch Tensors from CVAE outputs.

        Args:
            predictions: A dictionary containing:
                         - 'recon_x': Reconstructed input (Tensor).
                         - 'mu': Mean of the latent distribution (Tensor).
                         - 'logvar': Log-variance of the latent distribution (Tensor).
            targets: The true target values (original input X, np.ndarray or torch.Tensor).

        Returns:
            torch.Tensor: The computed ELBO loss as a scalar PyTorch Tensor.
        """
        if not isinstance(predictions, dict):
            raise TypeError(
                "ELBOLossCalculator expects 'predictions' to be a dictionary."
            )
        if not all(key in predictions for key in ["recon_x", "mu", "logvar"]):
            raise ValueError(
                "ELBOLossCalculator 'predictions' dict must contain 'recon_x', 'mu', and 'logvar' keys."
            )
        if not all(
            isinstance(predictions[key], Tensor) for key in ["recon_x", "mu", "logvar"]
        ):
            raise TypeError(
                "All values in 'predictions' dict must be torch.Tensor instances."
            )

        recon_x = predictions["recon_x"]
        mu = predictions["mu"]
        logvar = predictions["logvar"]

        # Ensure targets_tensor is a Tensor and on the same device as recon_x
        targets_tensor = to_tensor(targets, dtype=recon_x.dtype, device=recon_x.device)

        # Calculate Reconstruction Loss (Mean Squared Error)
        # Using reduction='sum' and then dividing by batch_size for consistency with KL
        # Or, just use 'mean' for both if that's desired behavior.
        # Let's stick to 'mean' as it's common and what F.mse_loss(..., reduction='mean') does.
        recon_loss = F.mse_loss(recon_x, targets_tensor, reduction="mean")

        # Calculate KL Divergence Loss
        # KL(q || p) = 0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        # Divided by batch_size for mean KL divergence per sample
        kl_div = (
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / recon_x.size(0)
        )

        total_loss = self.recon_weight * recon_loss + self.kl_weight * kl_div
        return total_loss
