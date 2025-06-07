import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from ...utils.similarities import SimilarityMethod
from ..domain.preference import ObjectivePreferences
from .base import BaseInterpolator


class Decoder(nn.Module):
    """
    Neural network decoder that maps a scalar interpolation coordinate (alpha) to a high-dimensional solution vector.

    Args:
        input_dim (int): Dimension of the output solution vector.
        hidden_dim (int): Number of hidden units in the hidden layer. Default is 64.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),  # Input: scalar alpha
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),  # Output: solution vector
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            alpha (torch.Tensor): Tensor of shape (batch_size, 1) representing interpolation coordinates.

        Returns:
            torch.Tensor: Predicted solution vectors of shape (batch_size, input_dim).
        """
        return self.net(alpha)

    def prepare_tensors(
        self,
        alphas: np.ndarray,
        pareto_set: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Converts input arrays to torch tensors and moves them to the proper device.

        Args:
            alphas (np.ndarray): Array of shape (N,) with scalar interpolation coordinates.
            pareto_set (np.ndarray): Array of shape (N, input_dim) with corresponding solutions.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of (alphas_tensor, pareto_tensor) ready for training.
        """
        alphas_tensor = (
            torch.tensor(alphas, dtype=torch.float32).unsqueeze(1).to(self.device)
        )
        pareto_tensor = torch.tensor(pareto_set, dtype=torch.float32).to(self.device)
        return alphas_tensor, pareto_tensor

    def train_model(
        self,
        alphas: np.ndarray,
        pareto_set: np.ndarray,
        epochs: int = 1000,
        lr: float = 0.001,
        verbose: bool = False,
    ) -> list[float]:
        """
        Train the decoder to map interpolation coordinates to solution vectors.

        Args:
            alphas (np.ndarray): Interpolation coordinates, shape (N,).
            pareto_set (np.ndarray): Corresponding solution vectors, shape (N, input_dim).
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
            verbose (bool): If True, print training progress.

        Returns:
            list[float]: list of training losses per epoch.
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        alphas_tensor, pareto_tensor = self.prepare_tensors(alphas, pareto_set)

        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(alphas_tensor)
            loss = loss_fn(outputs, pareto_tensor)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

        return losses

    def predict(self, alpha: float) -> np.ndarray:
        """
        Predict the solution vector for a single interpolation coordinate.

        Args:
            alpha (float): Scalar interpolation coordinate in [0, 1].

        Returns:
            np.ndarray: Predicted solution vector of shape (input_dim,).
        """
        self.eval()
        with torch.no_grad():
            input_tensor = torch.tensor([[alpha]], dtype=torch.float32).to(self.device)
            output = self(input_tensor)
            return output.cpu().numpy().flatten()


class NeuralNetworkInterpolator(BaseInterpolator):
    """
    Interpolator that uses a neural network decoder to map interpolation parameters (alpha)
    to decision vectors.

    This interpolator fits a neural network on given candidate solutions indexed by parametric
    coordinates (alphas). During recommendation, it computes a weighted alpha from user preferences
    and predicts the decision vector using the trained network.
    """

    def __init__(
        self,
        decoder: Decoder,
        similarity_metric: SimilarityMethod,
        epochs: int = 1000,
        learning_rate: float = 0.001,
    ) -> None:
        self.decoder = decoder
        self.similarity_metric = similarity_metric
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.param_coords: NDArray[np.float64] | None = None
        self.candidate_solutions: NDArray[np.float64] | None = None
        self.objective_front: NDArray[np.float64] | None = None

    def fit(
        self,
        candidate_solutions: NDArray[np.float64],
        objective_front: NDArray[np.float64],
    ) -> None:
        """
        Fit the neural network decoder on candidate solutions indexed by parametric coordinates.

        Args:
            candidate_solutions: Array of shape (N, D) in decision space.
            objective_front: Array of shape (N, M) in objective space.
        """
        if len(candidate_solutions) != len(objective_front):
            raise ValueError(
                "candidate_solutions and objective_front must have same length"
            )

        # Store data
        self.candidate_solutions = candidate_solutions
        self.objective_front = objective_front

        # Compute parametric coordinates: simple linear param based on first objective
        if len(candidate_solutions) == 1:
            self.param_coords = np.array([0.5])
        else:
            sorted_idx = np.argsort(objective_front[:, 0])
            self.param_coords = np.linspace(0.0, 1.0, len(candidate_solutions))[
                sorted_idx
            ]

        # Sort candidate_solutions accordingly
        self.candidate_solutions = candidate_solutions[sorted_idx]

        # Train the decoder neural network
        self.decoder.train_model(
            alphas=self.param_coords,
            pareto_set=self.candidate_solutions,
            epochs=self.epochs,
            lr=self.learning_rate,
            verbose=False,
        )

    def generate(self, preferences: ObjectivePreferences) -> NDArray[np.float64]:
        """
        Generate a decision vector based on user preferences.

        Args:
            preferences: User objective preferences.

        Returns:
            Decision vector as numpy array.
        """
        if self.param_coords is None or self.candidate_solutions is None:
            raise ValueError("Interpolator not fitted yet.")

        # Compute similarity scores between objectives and user weights
        weights = np.array([preferences.time_weight, preferences.energy_weight])
        similarity_scores = self.similarity_metric(self.objective_front, weights)
        total_similarity = np.sum(similarity_scores)

        # Compute weighted interpolation coordinate alpha
        if abs(total_similarity) < 1e-12:
            alpha = float(np.mean(self.param_coords))
        else:
            alpha = float(
                np.dot(self.param_coords, similarity_scores) / total_similarity
            )

        # Predict decision vector using neural network decoder
        return self.decoder.predict(alpha)
