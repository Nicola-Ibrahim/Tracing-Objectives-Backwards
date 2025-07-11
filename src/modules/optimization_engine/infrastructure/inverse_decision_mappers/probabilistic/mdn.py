import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

from ....domain.interpolation.interfaces.base_inverse_decision_mapper import (
    BaseInverseDecisionMapper,
)


class MDN(nn.Module):
    """
    A Mixture Density Network (MDN) module.

    Args:
        input_dim (int): The dimensionality of the input features.
        output_dim (int): The dimensionality of the output (target) features.
        num_mixtures (int): The number of Gaussian mixture components.
    """

    def __init__(self, input_dim: int, output_dim: int, num_mixtures: int = 5):
        super().__init__()
        hidden_size = 64
        # Fully connected layer for initial feature transformation
        self.fc1 = nn.Linear(input_dim, hidden_size)
        # Layer to predict the mixing coefficients (pi) for each mixture component
        self.fc_pi = nn.Linear(hidden_size, num_mixtures)
        # Layer to predict the mean (mu) for each mixture component
        self.fc_mu = nn.Linear(hidden_size, num_mixtures * output_dim)
        # Layer to predict the standard deviation (sigma) for each mixture component
        # We predict log(sigma) to ensure sigma is positive, then exponentiate
        self.fc_sigma = nn.Linear(hidden_size, num_mixtures * output_dim)

        self.output_dim = output_dim
        self.num_mixtures = num_mixtures

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the MDN.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - pi (torch.Tensor): The mixing coefficients for each mixture component,
                                     normalized using softmax.
                - mu (torch.Tensor): The means of the Gaussian mixture components.
                - sigma (torch.Tensor): The standard deviations of the Gaussian mixture components.
        """
        # Apply ReLU activation to the output of the first fully connected layer
        h = F.relu(self.fc1(x))
        # Calculate mixing coefficients (pi) and apply softmax for normalization
        pi = F.softmax(self.fc_pi(h), dim=1)
        # Calculate means (mu) and reshape to (batch_size, num_mixtures, output_dim)
        mu = self.fc_mu(h).view(-1, self.num_mixtures, self.output_dim)
        # Calculate standard deviations (sigma) by exponentiating the output
        # and reshape to (batch_size, num_mixtures, output_dim)
        sigma = torch.exp(self.fc_sigma(h)).view(-1, self.num_mixtures, self.output_dim)
        return pi, mu, sigma


class MDNInverseDecisionMapper(BaseInverseDecisionMapper):
    """
    An inverse mapper that uses a Mixture Density Network (MDN) to model the
    inverse relationship from objectives to decisions.

    This mapper learns a conditional probability distribution p(decisions | objectives),
    allowing for the prediction of a distribution of decisions for given objectives,
    rather than a single deterministic decision.
    """

    def __init__(
        self, num_mixtures: int = 5, epochs: int = 500, learning_rate: float = 1e-3
    ):
        """
        Initializes the MDNInverseMapper.

        Args:
            num_mixtures (int): The number of Gaussian mixture components for the MDN.
            epochs (int): The number of training epochs.
            lr (float): The learning rate for the Adam optimizer.
        """
        super().__init__()
        self._num_mixtures = num_mixtures
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._model: MDN | None = None  # Initialize model as None, to be set in fit

    def fit(
        self, objectives: NDArray[np.float64], decisions: NDArray[np.float64]
    ) -> None:
        """
        Fits the MDN model to the provided objectives and decisions.

        Args:
            objectives (NDArray[np.float64]): The input objective values (features X).
            decisions (NDArray[np.float64]): The target decision values (targets Y).
        """
        super().fit(objectives, decisions)  # Call parent class fit method if necessary

        # Convert numpy arrays to PyTorch tensors
        X = torch.tensor(objectives, dtype=torch.float32)
        Y = torch.tensor(decisions, dtype=torch.float32)

        # Initialize the MDN model with appropriate input and output dimensions
        self._model = MDN(
            input_dim=X.shape[1], output_dim=Y.shape[1], num_mixtures=self._num_mixtures
        )
        # Initialize the Adam optimizer
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)

        # Training loop
        for epoch in range(self._epochs):
            optimizer.zero_grad()  # Zero the gradients
            pi, mu, sigma = self._model(X)  # Forward pass
            # Create a MixtureSameFamily distribution
            # The Categorical distribution models the mixture coefficients (pi)
            # The Independent(Normal) models the individual Gaussian components (mu, sigma)
            dist = MixtureSameFamily(Categorical(pi), Independent(Normal(mu, sigma), 1))
            # Calculate the negative log-likelihood loss
            # We want to maximize the likelihood, so we minimize the negative log-likelihood
            loss = -dist.log_prob(Y).mean()
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters

    def predict(self, target_objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predicts decisions for given target objectives by sampling from the learned
        mixture distribution.

        Args:
            target_objectives (NDArray[np.float64]): The objectives for which to predict decisions.

        Returns:
            NDArray[np.float64]: The sampled decisions corresponding to the target objectives.
        """
        if self._model is None:
            raise RuntimeError("The model has not been fit yet. Call 'fit' first.")

        self._model.eval()  # Set the model to evaluation mode
        X = torch.tensor(target_objectives, dtype=torch.float32)

        with torch.no_grad():  # Disable gradient calculation for inference
            pi, mu, sigma = self._model(
                X
            )  # Forward pass to get distribution parameters
            # Recreate the MixtureSameFamily distribution
            dist = MixtureSameFamily(Categorical(pi), Independent(Normal(mu, sigma), 1))
            samples = dist.sample()  # Sample decisions from the learned distribution

        return samples.numpy()  # Convert sampled decisions back to a numpy array
