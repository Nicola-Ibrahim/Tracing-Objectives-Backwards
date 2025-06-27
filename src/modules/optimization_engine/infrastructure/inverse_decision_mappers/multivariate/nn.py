# Neural Network Mapper
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from ....domain.interpolation.interfaces.base_inverse_decision_mappers import (
    BaseInverseDecisionMapper,
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

    def forward(self, objectives: torch.Tensor) -> torch.Tensor:
        return self.net(objectives)

    def prepare_tensors(
        self, objectives: np.ndarray, decisions: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        objectives_tensor = torch.tensor(objectives, dtype=torch.float32).to(
            self.device
        )
        decisions_tensor = torch.tensor(decisions, dtype=torch.float32).to(self.device)
        return objectives_tensor, decisions_tensor

    def train_model(
        self,
        objectives: np.ndarray,
        decisions: np.ndarray,
        epochs: int = 1000,
        lr: float = 0.001,
        verbose: bool = False,
    ) -> list[float]:
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        objectives_tensor, decisions_tensor = self.prepare_tensors(
            objectives, decisions
        )
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(objectives_tensor)
            loss = loss_fn(outputs, decisions_tensor)
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


class NeuralNetworkInterpolator(BaseInverseDecisionMapper):
    """
    Interpolator that uses a neural network decoder to map directly from
    an objective space to a decision space.
    """

    def __init__(
        self,
        epochs: int = 1000,
        learning_rate: float = 0.001,
    ) -> None:
        # The decoder is part of this class's state
        self.decoder = Decoder()
        self.epochs = epochs
        self.learning_rate = learning_rate
        # Call the parent's constructor to initialize shared state
        super().__init__()

    def fit(
        self,
        objectives: NDArray[np.float64],
        decisions: NDArray[np.float64],
    ) -> None:
        """
        Fits the neural network decoder to learn the direct mapping from objectives to decisions.
        """
        # Call the parent's fit to perform universal data validation and store dimensions.
        super().fit(objectives, decisions)

        # Check if the decoder's input/output dimensions match the data.
        if (
            self.decoder.net[0].in_features != self._objective_dim
            or self.decoder.net[-1].out_features != self._decision_dim
        ):
            raise ValueError(
                f"Decoder dimensions do not match data. Decoder expects input={self.decoder.net[0].in_features}, output={self.decoder.net[-1].out_features}. "
                f"Data has objectives={self._objective_dim}, decisions={self._decision_dim}."
                "Please instantiate the Decoder with matching dimensions before fitting."
            )

        # Train the decoder using the provided objectives and decisions
        self.decoder.train_model(
            objectives=objectives,
            decisions=decisions,
            epochs=self.epochs,
            lr=self.learning_rate,
            verbose=False,
        )

    def predict(self, target_objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predicts decision vectors for given target objective points using the trained network.
        """
        # Perform validation specific to this method.
        # This check is crucial and must be here because this method is independent of the fit method
        # for validation purposes.
        if self._objective_dim is None:
            raise RuntimeError("Mapper has not been fitted yet. Call fit() first.")

        if target_objectives.ndim == 1:
            target_objectives = target_objectives.reshape(-1, 1)

        if target_objectives.shape[1] != self._objective_dim:
            raise ValueError(
                f"Target objectives must have {self._objective_dim} dimensions, "
                f"but got {target_objectives.shape[1]} dimensions."
            )

        # Use the decoder to predict for the given target objectives
        return self.decoder.predict(target_objectives)
