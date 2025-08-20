import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ...domain.model_management.interfaces.base_forward_decision_mapper import (
    BaseForwardDecisionMapper,
)


class NeuralNetworkModel(nn.Module):
    """
    A modular PyTorch Neural Network model.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layer_dims: list = None,
        activation: nn.Module = nn.ReLU,
    ):
        """
        Initializes the NeuralNetworkModel.

        Args:
            input_dim (int): Dimensionality of the input.
            output_dim (int): Dimensionality of the output.
            hidden_layer_dims (list): A list of integers specifying the number of
                                  neurons in each hidden layer. If None, a
                                  default structure is used.
            activation (torch.nn.Module): The activation function to use for hidden layers.
        """
        super().__init__()

        self.activation_fn = activation()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layer_dims = (
            hidden_layer_dims if hidden_layer_dims is not None else [64, 32]
        )

        # Dynamically create layers and register them using ModuleList
        layers = []
        prev_dim = input_dim
        for i, h_dim in enumerate(self.hidden_layer_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim
        self.hidden_layers = nn.ModuleList(layers)

        self.output_layer = nn.Linear(prev_dim, output_dim)

        print(
            f"NeuralNetworkModel initialized with input_dim={input_dim}, "
            f"output_dim={output_dim}, hidden_layer_dims={self.hidden_layer_dims}."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for layer in self.hidden_layers:
            x = self.activation_fn(layer(x))
        x = self.output_layer(
            x
        )  # Output layer has no activation for regression by default
        return x


class NNForwardDecisionMapper(BaseForwardDecisionMapper):
    """
    An implementation of a forward mapper using a trainable PyTorch Neural Network.
    This mapper expects and returns numpy arrays at its public interface.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layer_dims: list = None,
        activation: nn.Module = nn.ReLU,
    ):
        """
        Initializes the NNForwardDecisionMapper.

        Args:
            input_dim (int): The dimension of the input decision.
            output_dim (int): The dimension of the output objective.
            hidden_layer_dims (list): A list of integers specifying the number of
                                  neurons in each hidden layer.
            activation (torch.nn.Module): The activation function for hidden layers.
        """
        self.model = NeuralNetworkModel(
            input_dim, output_dim, hidden_layer_dims, activation
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"NNForwardDecisionMapper model moved to: {self.device}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts y using the trained PyTorch Neural Network.

        Args:
            X: The input decision (numpy array).

        Returns:
            The predicted y (numpy array).
        """
        # Convert numpy array to torch.Tensor and move to device
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        # Add batch dimension if a single sample is provided
        if X_tensor.ndim == 1:
            X_tensor = X_tensor.unsqueeze(0)

        # Ensure model is in evaluation mode for prediction
        self.model.eval()
        with torch.no_grad():
            predicted_objectives_tensor = self.model(X_tensor)

        # Convert result back to numpy array and ensure float32
        return predicted_objectives_tensor.cpu().numpy().astype(np.float32)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        optimizer_cls=optim.Adam,
        loss_fn=nn.MSELoss(),
    ):
        """
        Trains the neural network model.

        Args:
            X: Training data for X (numpy array).
            y: Training data for y (numpy array).
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for the optimizer.
            optimizer_cls: The PyTorch optimizer class (e.g., torch.optim.Adam).
            loss_fn: The PyTorch loss function (e.g., torch.nn.MSELoss()).
        """
        # Convert numpy arrays to torch.Tensor and move to device
        decisions_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        objectives_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        # Ensure objectives_tensor has the correct shape (e.g., (N, 1) for scalar outputs)
        if objectives_tensor.ndim == 1 and self.model.output_dim == 1:
            objectives_tensor = objectives_tensor.unsqueeze(1)
        elif (
            objectives_tensor.ndim == 2
            and objectives_tensor.shape[1] != self.model.output_dim
        ):
            raise ValueError(
                f"objectives_tensor second dimension ({objectives_tensor.shape[1]}) "
                f"does not match model output_dim ({self.model.output_dim})"
            )

        dataset = TensorDataset(decisions_tensor, objectives_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optimizer_cls(self.model.parameters(), lr=learning_rate)

        self.model.train()  # Set model to training mode
        print(f"\nStarting training for NNForwardDecisionMapper for {epochs} epochs...")
        for epoch in tqdm(range(epochs), desc="Training Progress"):
            total_loss = 0
            for batch_decisions, batch_objectives in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_decisions)
                loss = loss_fn(outputs, batch_objectives)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Optional: Print loss every few epochs
            if (epoch + 1) % (
                epochs // 10 if epochs >= 10 else 1
            ) == 0 or epoch == epochs - 1:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        self.model.eval()  # Set model to evaluation mode after training
        print("Training complete.")
