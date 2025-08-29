import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

from ....domain.model_management.interfaces.base_estimator import (
    ProbabilisticEstimator,
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

    @property
    def type(self) -> str:
        return "CVAE"

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


class CVAEEstimator(ProbabilisticEstimator):
    """
    An inverse mapper that uses a Conditional Variational Autoencoder (CVAE)
    to model the inverse relationship from objectives to decisions.

    The CVAE learns a mapping from objectives (conditions) to a distribution
    over decisions, allowing for the generation of diverse decisions that
    are likely to achieve the target objectives.
    """

    def __init__(self, latent_dim: int = 8, learning_rate: float = 1e-3):
        """
        Initializes the CVAEInverseMapper.

        Args:
            latent_dim (int): Dimensionality of the latent space in the CVAE.
            epochs (int): The number of training epochs.
            learning_rate (float): The learning rate for the Adam optimizer.
        """
        super().__init__()
        self._latent_dim = latent_dim
        self._learning_rate = learning_rate
        self._encoder: CVAEEncoder | None = None
        self._decoder: CVAEDecoder | None = None
        self._y_dim: int | None = None  # Dimensionality of objectives (y)
        self._x_dim: int | None = None  # Dimensionality of decisions (x)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._training_history = {"epochs": [], "recon_loss": [], "kl": []}

    @property
    def type(self) -> str:
        return "Conditional VAE"

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64], **kwargs) -> None:
        """
        Fits the CVAE model to the provided training features (X) and targets (y).

        The CVAE is trained to reconstruct targets (y) conditioned on features (X),
        while also ensuring the latent space follows a prior distribution.

        Args:
            X (NDArray[np.float64]): Training input features.
            y (NDArray[np.float64]): Training targets.
        """
        super().fit(X, y)  # Call parent class fit method

        # Unpack additional keyword arguments
        epochs = kwargs.get("epochs", 100)
        batch_size = kwargs.get("batch_size", 64)

        # Determine the dimensions of the objective and decision spaces
        self._y_dim = X.shape[1]
        self._x_dim = y.shape[1]

        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(y, dtype=torch.float32)

        # Initialize the encoder and decoder modules
        self._encoder = CVAEEncoder(self._y_dim, self._x_dim, self._latent_dim)
        self._decoder = CVAEDecoder(self._latent_dim, self._y_dim, self._x_dim)

        # Initialize the Adam optimizer for both encoder and decoder parameters
        optimizer = torch.optim.Adam(
            list(self._encoder.parameters()) + list(self._decoder.parameters()),
            lr=self._learning_rate,
        )
        n = X_tensor.shape[0]

        # Training loop
        for epoch in range(1, epochs + 1):
            self._encoder.train()
            self._decoder.train()
            perm = torch.randperm(n)
            epoch_recon = 0.0
            epoch_kl = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                yc = X_tensor[idx]  # condition
                xt = Y_tensor[idx]  # target to reconstruct
                optimizer.zero_grad()
                mu, logvar = self._encoder(
                    xt
                )  # note: your earlier code used Y as input; many CVAEs condition differently; adopt your desired mapping
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
                recon = self._decoder(
                    z, yt := yc
                )  # reconstruct decisions given z and conditioning
                recon_loss = F.mse_loss(recon, xt, reduction="mean")
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl
                loss.backward()
                optimizer.step()
                epoch_recon += float(recon_loss.detach().cpu().item()) * len(idx)
                epoch_kl += float(kl.detach().cpu().item()) * len(idx)
            epoch_recon /= n
            epoch_kl /= n
            self._training_history["epochs"].append(epoch)
            self._training_history["recon_loss"].append(epoch_recon)
            self._training_history["kl"].append(epoch_kl)

    def predict(
        self, X: NDArray[np.float64], n_samples: int = 1
    ) -> NDArray[np.float64]:
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

        # Convert input features to PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            batch = X_tensor.shape[0]
            z = torch.randn(
                batch, n_samples, self.latent_dim, device=self.device
            )  # (batch, n_samples, latent)
            # decode per sample
            outs = []
            for s in range(n_samples):
                zs = z[:, s, :]
                out = self._decoder(zs, X_tensor)  # (batch, x_dim)
                outs.append(out.unsqueeze(1))
            outs = torch.cat(outs, dim=1)  # (batch, n_samples, x_dim)
            if n_samples == 1:
                return outs[:, 0, :].cpu().numpy()
            else:
                return outs.cpu().numpy()
