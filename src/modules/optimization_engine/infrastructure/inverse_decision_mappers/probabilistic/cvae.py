import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

from ....domain.interpolation.interfaces.base_inverse_decision_mappers import (
    BaseInverseDecisionMapper,
)


class CVAEEncoder(nn.Module):
    """
    Encoder module for the Conditional Variational Autoencoder (CVAE).

    This encoder takes an objective (y) as input and outputs the parameters
    (mean and log-variance) of a Gaussian distribution in the latent space.
    """

    def __init__(self, y_dim: int, x_dim: int, latent_dim: int = 8):
        """
        Initializes the CVAEEncoder.

        Args:
            y_dim (int): Dimensionality of the objective space (condition).
            x_dim (int): Dimensionality of the decision space (input to decoder, not directly used here but for context).
            latent_dim (int): Dimensionality of the latent space.
        """
        super().__init__()
        # First fully connected layer for initial processing of the objective (y)
        self.fc1 = nn.Linear(y_dim, 32)
        # Fully connected layer to output the mean of the latent distribution
        self.fc21 = nn.Linear(32, latent_dim)
        # Fully connected layer to output the log-variance of the latent distribution
        self.fc22 = nn.Linear(32, latent_dim)

    def forward(self, y: torch.Tensor):
        """
        Forward pass of the encoder.

        Args:
            y (torch.Tensor): The objective tensor, representing the condition.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - mu (torch.Tensor): The mean of the latent distribution.
                - logvar (torch.Tensor): The log-variance of the latent distribution.
        """
        # Apply ReLU activation to the output of the first fully connected layer
        h = F.relu(self.fc1(y))
        # Return the mean and log-variance
        return self.fc21(h), self.fc22(h)


class CVAEDecoder(nn.Module):
    """
    Decoder module for the Conditional Variational Autoencoder (CVAE).

    This decoder takes a latent space sample (z) and an objective (y) as input
    and reconstructs the decision (x).
    """

    def __init__(self, latent_dim: int, y_dim: int, x_dim: int):
        """
        Initializes the CVAEDecoder.

        Args:
            latent_dim (int): Dimensionality of the latent space.
            y_dim (int): Dimensionality of the objective space (condition).
            x_dim (int): Dimensionality of the decision space (output).
        """
        super().__init__()
        # First fully connected layer, taking concatenated latent sample and objective
        self.fc1 = nn.Linear(latent_dim + y_dim, 32)
        # Second fully connected layer to output the reconstructed decision
        self.fc2 = nn.Linear(32, x_dim)

    def forward(self, z: torch.Tensor, y: torch.Tensor):
        """
        Forward pass of the decoder.

        Args:
            z (torch.Tensor): A sample from the latent space.
            y (torch.Tensor): The objective tensor, representing the condition.

        Returns:
            torch.Tensor: The reconstructed decision tensor.
        """
        # Concatenate the latent sample (z) and the objective (y)
        # Apply ReLU activation to the output of the first fully connected layer
        h = F.relu(self.fc1(torch.cat([z, y], dim=1)))
        # Return the reconstructed decision
        return self.fc2(h)


class CVAEInverseDecisionMapper(BaseInverseDecisionMapper):
    """
    An inverse mapper that uses a Conditional Variational Autoencoder (CVAE)
    to model the inverse relationship from objectives to decisions.

    The CVAE learns a mapping from objectives (conditions) to a distribution
    over decisions, allowing for the generation of diverse decisions that
    are likely to achieve the target objectives.
    """

    def __init__(self, latent_dim: int = 8, epochs: int = 500, lr: float = 1e-3):
        """
        Initializes the CVAEInverseMapper.

        Args:
            latent_dim (int): Dimensionality of the latent space in the CVAE.
            epochs (int): The number of training epochs.
            lr (float): The learning rate for the Adam optimizer.
        """
        super().__init__()
        self._latent_dim = latent_dim
        self._epochs = epochs
        self._lr = lr
        self._encoder: CVAEEncoder | None = None
        self._decoder: CVAEDecoder | None = None
        self._y_dim: int | None = None  # Dimensionality of objectives (y)
        self._x_dim: int | None = None  # Dimensionality of decisions (x)

    def fit(
        self, objectives: NDArray[np.float64], decisions: NDArray[np.float64]
    ) -> None:
        """
        Fits the CVAE model to the provided objectives and decisions.

        The CVAE is trained to reconstruct decisions (X) conditioned on objectives (Y),
        while also ensuring the latent space follows a prior distribution (e.g., a standard normal).

        Args:
            objectives (NDArray[np.float64]): The input objective values (conditions Y).
            decisions (NDArray[np.float64]): The target decision values (data X).
        """
        super().fit(objectives, decisions)  # Call parent class fit method if necessary

        # Determine the dimensions of the objective and decision spaces
        self._y_dim = objectives.shape[1]
        self._x_dim = decisions.shape[1]

        # Convert numpy arrays to PyTorch tensors
        X = torch.tensor(decisions, dtype=torch.float32)  # Decisions are X
        Y = torch.tensor(
            objectives, dtype=torch.float32
        )  # Objectives are Y (conditions)

        # Initialize the encoder and decoder modules
        self._encoder = CVAEEncoder(self._y_dim, self._x_dim, self._latent_dim)
        self._decoder = CVAEDecoder(self._latent_dim, self._y_dim, self._x_dim)

        # Initialize the Adam optimizer for both encoder and decoder parameters
        optimizer = torch.optim.Adam(
            list(self._encoder.parameters()) + list(self._decoder.parameters()),
            lr=self._lr,
        )

        # Training loop
        for epoch in range(self._epochs):
            optimizer.zero_grad()  # Zero the gradients

            # Encoder forward pass: get mean and log-variance of the latent distribution
            mu, logvar = self._encoder(Y)

            # Reparameterization trick: sample z from the latent distribution
            std = torch.exp(
                0.5 * logvar
            )  # Calculate standard deviation from log-variance
            z = mu + std * torch.randn_like(std)  # Sample z

            # Decoder forward pass: reconstruct X from z and Y
            recon_x = self._decoder(z, Y)

            # Calculate Reconstruction Loss (Mean Squared Error)
            recon_loss = F.mse_loss(recon_x, X)

            # Calculate KL Divergence Loss
            # KL divergence between the learned latent distribution q(z|x,y)
            # and the prior p(z) (assumed to be a standard normal distribution)
            # The formula for KL divergence for diagonal Gaussians is:
            # KL(q || p) = 0.5 * sum(1 + logvar - mu^2 - exp(logvar))
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / X.size(0)

            # Total loss is the sum of reconstruction loss and KL divergence
            loss = recon_loss + kl_div
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters

    def predict(self, target_objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Generates decisions for given target objectives by sampling from the CVAE's decoder.

        During prediction, we sample a random vector from a standard normal distribution
        (representing the prior in the latent space) and condition the decoder on the
        target objectives to generate new decisions.

        Args:
            target_objectives (NDArray[np.float64]): The objectives for which to generate decisions.

        Returns:
            NDArray[np.float64]: The generated decisions corresponding to the target objectives.
        """
        if self._encoder is None or self._decoder is None:
            raise RuntimeError("The model has not been fit yet. Call 'fit' first.")
        if self._latent_dim is None:
            raise RuntimeError("Latent dimension not set. Call 'fit' first.")

        self._encoder.eval()  # Set encoder to evaluation mode
        self._decoder.eval()  # Set decoder to evaluation mode

        # Convert target objectives to PyTorch tensor
        Y = torch.tensor(target_objectives, dtype=torch.float32)

        with torch.no_grad():  # Disable gradient calculation for inference
            # Sample a random vector from a standard normal distribution for the latent space
            # The number of samples corresponds to the number of target objectives
            z = torch.randn(Y.shape[0], self._latent_dim)
            # Decode the latent sample and the target objectives to generate decisions
            X_pred = self._decoder(z, Y)

        # Convert the predicted decisions back to a numpy array
        return X_pred.numpy()
