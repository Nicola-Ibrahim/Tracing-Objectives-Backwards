# Neural Network Mapper
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from ....domain.model_management.interfaces.base_inverse_decision_mapper import (
    DeterministicInverseDecisionMapper,
)


class Decoder(nn.Module):
    def __init__(
        self, objective_dim: int, decision_dim: int, hidden_dim: int = 64
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(objective_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, decision_dim),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)

    def prepare_tensors(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        return X_tensor, y_tensor

    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        lr: float = 0.001,
        verbose: bool = False,
    ) -> list[float]:
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        X_tensor, y_tensor = self.prepare_tensors(X, y)
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(X_tensor)
            loss = loss_fn(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")
        return losses

    def predict(self, targets: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(targets, dtype=torch.float32).to(self.device)
            output = self(input_tensor)
            return output.cpu().numpy()


class NNInverseDecisionMapper(DeterministicInverseDecisionMapper):
    """
    Interpolator that uses a neural network decoder to map directly from
    an objective space to a decision space.
    """

    def __init__(
        self,
        objective_dim: int = 2,
        decision_dim: int = 2,
        epochs: int = 1000,
        learning_rate: float = 0.001,
    ) -> None:
        # The decoder is part of this class's state
        self.decoder = Decoder(objective_dim=objective_dim, decision_dim=decision_dim)
        self.epochs = epochs
        self.learning_rate = learning_rate
        # Call the parent's constructor to initialize shared state
        super().__init__()

    @property
    def type(self) -> str:
        return "NNInverseDecisionMapper"

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """
        Fits the neural network decoder to learn the direct mapping from X to y.
        """
        # Call the parent's fit to perform universal data validation and store dimensions.
        super().fit(X, y)

        # Check if the decoder's input/output dimensions match the data.
        if (
            self.decoder.net[0].in_features != self._objective_dim
            or self.decoder.net[-1].out_features != self._decision_dim
        ):
            raise ValueError(
                f"Decoder dimensions do not match data. Decoder expects input={self.decoder.net[0].in_features}, output={self.decoder.net[-1].out_features}. "
                f"Data has inputs={self._objective_dim}, outputs={self._decision_dim}."
                "Please instantiate the Decoder with matching dimensions before fitting."
            )

        # Train the decoder using the provided X and y
        self.decoder.train_model(
            X, y, epochs=self.epochs, lr=self.learning_rate, verbose=False
        )

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predicts decision vectors for given input feature points using the trained network.
        """
        if self._objective_dim is None:
            raise RuntimeError("Mapper has not been fitted yet. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self._objective_dim:
            raise ValueError(
                f"Input must have {self._objective_dim} dimensions, but got {X.shape[1]} dimensions."
            )

        return self.decoder.predict(X)
