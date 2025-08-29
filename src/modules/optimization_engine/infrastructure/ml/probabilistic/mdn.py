from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from torch.distributions import (
    Categorical,
    Gamma,
    Independent,
    Laplace,
    LogNormal,
    MixtureSameFamily,
    Normal,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from umap import UMAP

from ....domain.model_management.interfaces.base_estimator import (
    ProbabilisticEstimator,
)


@dataclass
class TrainingHistory:
    # small container, easy to serialize as dict(training_history.__dict__)
    epochs: list
    train_loss: list
    val_loss: list


class ActivationFunctionEnum(Enum):
    """An Enum for available activation function names."""

    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SOFTPLUS = "softplus"
    IDENTITY = "identity"


class DistributionFamilyEnum(Enum):
    """An Enum for available distribution families."""

    NORMAL = "normal"
    LAPLACE = "laplace"
    LOGNORMAL = "lognormal"


class OptimizerFunctionEnum(Enum):
    """An Enum for available optimizers."""

    SGD = "sgd"
    ADAM = "adam"
    RMSPROP = "rmsprop"
    GRADIENT_DESCENT = "gradient_descent"


class MDN(nn.Module):
    """
    A Mixture Density Network (MDN) module with configurable architecture.

        Architecture Overview:
        Input (input_dim)
           │
           ▼
    ┌─────────────────────────────┐
    │  Hidden Stack (configurable)│
    │  e.g., Linear → ReLU → ...  │
    └─────────────────────────────┘
           │
           ▼
    Final Hidden Representation (H)
           │
           ├──> π-head:  Linear(H → num_mixtures)
           │              → Softmax over mixtures
           │
           ├──> μ-head:  Linear(H → num_mixtures*output_dim)
           │              → reshape to (num_mixtures, output_dim)
           │
           └──> σ-head:  Linear(H → num_mixtures*output_dim)
                          → Softplus (or exp) for positivity
                          → reshape to (num_mixtures, output_dim)

    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_mixtures: int = 5,
        hidden_layers: list[int] = [64, 32],
        hidden_activation_fn_name: ActivationFunctionEnum = ActivationFunctionEnum.RELU,
        pi_bias_init: list | None = None,
        mu_bias_init: list | None = None,
    ):
        """
        Initialize the Mixture Density Network (MDN).
        """
        super().__init__()

        self.num_mixtures = num_mixtures
        self.output_dim = output_dim

        # Get activation function
        self.hidden_activation_fn_name = self._get_activation(hidden_activation_fn_name)

        # ----- Build hidden stack -----
        layers = []
        in_size = input_dim
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(self.hidden_activation_fn_name)
            in_size = hidden_size

        if layers:  # remove last activation to let heads apply their own logic
            layers.pop()

        self.hidden_stack = nn.Sequential(*layers)

        final_hidden_size = hidden_layers[-1] if hidden_layers else input_dim

        # ----- Output heads -----
        # Mixture weights (π) → Softmax later
        self.fc_pi = nn.Linear(final_hidden_size, num_mixtures)
        nn.init.normal_(self.fc_pi.weight)
        if pi_bias_init is not None:
            nn.init.constant_(self.fc_pi.bias, pi_bias_init)

        # Means (μ) → shape (num_mixtures, output_dim)
        self.fc_mu = nn.Linear(final_hidden_size, num_mixtures * output_dim)
        if mu_bias_init is not None:
            nn.init.constant_(self.fc_mu.bias, mu_bias_init)

        # Standard deviations (σ) → positive via softplus/exp
        self.fc_sigma = nn.Linear(final_hidden_size, num_mixtures * output_dim)
        nn.init.normal_(self.fc_sigma.weight)

    # ------- Utility Functions -------
    def _get_activation(self, activation_fn: ActivationFunctionEnum):
        """
        Get the activation function based on the specified activation function name.
        """
        activation_fns = {
            ActivationFunctionEnum.RELU: nn.ReLU(),
            ActivationFunctionEnum.TANH: nn.Tanh(),
            ActivationFunctionEnum.SIGMOID: nn.Sigmoid(),
            ActivationFunctionEnum.SOFTPLUS: nn.Softplus(),
            ActivationFunctionEnum.IDENTITY: nn.Identity(),
        }

        return activation_fns.get(activation_fn, nn.Identity())

    def _get_distribution(
        self,
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        distribution_family: DistributionFamilyEnum,
    ) -> MixtureSameFamily:
        """
        Gets a `MixtureSameFamily` distribution object.
        """
        if distribution_family == DistributionFamilyEnum.NORMAL:
            dist = Normal(mu, sigma)
        elif distribution_family == DistributionFamilyEnum.LAPLACE:
            dist = Laplace(mu, sigma)
        elif distribution_family == DistributionFamilyEnum.LOGNORMAL:
            dist = LogNormal(mu, sigma)
        else:
            raise ValueError(
                f"Unsupported distribution family: {distribution_family.value}"
            )

        return MixtureSameFamily(Categorical(probs=pi), Independent(dist, 1))

    def summary(self, input_size=(1,)):
        """
        Prints a summary of the architecture.
        """
        # This part of the code is unchanged as it is not a critical part of the MDN logic.
        # It's a nice utility that should be kept as-is.
        x = torch.zeros(1, *input_size)
        print("\nMixture Density Network (MDN) Summary\n" + "=" * 50)

        layers = []
        layers.append(("Input", x.shape))

        tmp_x = x
        for i, layer in enumerate(
            tqdm(self.hidden_stack, desc="Building Summary", unit="layer")
        ):
            tmp_x = layer(tmp_x)
            layers.append((layer.__class__.__name__, tmp_x.shape))

        pi = F.softmax(self.fc_pi(tmp_x), dim=-1)
        mu = self.fc_mu(tmp_x).view(-1, self.num_mixtures, self.output_dim)
        sigma = F.softplus(self.fc_sigma(tmp_x)).view(
            -1, self.num_mixtures, self.output_dim
        )

        layers.append(("π (mixture weights)", pi.shape))
        layers.append(("μ (means)", mu.shape))
        layers.append(("σ (stddev)", sigma.shape))

        for name, shape in layers:
            print(f"{name:<25} │ Output Shape: {list(shape)}")
        print("=" * 50)

    # ----- Forward Pass -----

    def forward(self, x: torch.Tensor):
        # Apply activation to the output of the first fully connected layer
        # Shared hidden representation
        h = self.hidden_stack(x)  # [batch_size, final_hidden_size]

        # Calculate mixing coefficients (pi) and apply softmax for normalization
        pi = self.fc_pi(h)  # [batch_size, num_mixtures]
        pi = F.softmax(
            pi, dim=-1
        )  # turns real outputs into a valid discrete probability distribution (non-negative, sums to 1).

        # Calculate means (mu) and reshape to (batch_size, num_mixtures, output_dim)
        mu = self.fc_mu(h)  # [batch_size, num_mixtures * output_dim]
        mu = mu.view(-1, self.num_mixtures, self.output_dim)  # can be a real number

        # Calculate standard deviations (sigma) by exponentiating the output
        # Two problems:
        # 1) exp(large_negative) => extremely small sigma (e.g. 1e-50) -> (y-mu)^2 / sigma^2 huge -> overflow/NaN
        # 2) exp(large_positive) => huge sigma -> underflow/very flat Gaussian -> training might be unstable
        sigma = self.fc_sigma(h)  # [batch_size, num_mixtures * output_dim]
        sigma = (
            F.softplus(sigma) + 1e-6
        )  # Enforce positive stddev so the Gaussian is well-defined and gradients behave predictably
        sigma = sigma.view(-1, self.num_mixtures, self.output_dim)

        return pi, mu, sigma


class MDNEstimator(ProbabilisticEstimator):
    """
    An inverse mapper that uses a Mixture Density Network (MDN) to model the
    inverse relationship from objectives to decisions.
    """

    def __init__(
        self,
        num_mixtures: int = -1,
        learning_rate: float = 1e-3,
        early_stopping_patience: int = 10,
        distribution_family: DistributionFamilyEnum = DistributionFamilyEnum.NORMAL,
        gmm_boost: bool = False,
        hidden_layers: list[int] = [64],
        hidden_activation_fn_name: ActivationFunctionEnum = ActivationFunctionEnum.RELU,
        optimizer_fn_name: OptimizerFunctionEnum = OptimizerFunctionEnum.ADAM,
        verbose: bool = False,
    ):
        """
        Initialize the MDNEstimator.
        """
        super().__init__()
        self._num_mixtures = num_mixtures
        self._learning_rate = learning_rate
        self._early_stopping_patience = early_stopping_patience
        self._distribution_family = distribution_family
        self._gmm_boost = gmm_boost
        self._hidden_layers = hidden_layers
        self._hidden_activation_fn_name = hidden_activation_fn_name
        self._optimizer_fn_name = optimizer_fn_name
        self._verbose = verbose
        self._model: MDN | None = None
        self._clusterer: GaussianMixture | None = None
        self._best_model_state_dict: dict | None = None
        self._training_history: dict | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def type(self) -> str:
        return "MDN"

    def _determine_num_mixtures(self, X_y: npt.NDArray[np.float64]) -> int:
        lowest_bic = np.inf
        best_gmm = None
        n_components_range = range(1, 7)
        for n_components in n_components_range:
            gmm = GaussianMixture(
                n_components=n_components, covariance_type="full", max_iter=10000
            )
            gmm.fit(X_y)
            bic = gmm.bic(X_y)
            if bic < lowest_bic:
                lowest_bic = bic
                best_gmm = gmm
        self._clusterer = best_gmm
        return best_gmm.n_components

    def _prepare_model(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
        input_dim = X.shape[1]
        output_dim = y.shape[1]

        if self._num_mixtures == -1:
            self._num_mixtures = self._determine_num_mixtures(np.hstack((X, y)))

        if self._gmm_boost:
            cluster_probs = self._clusterer.predict_proba(X)
            X_tensor = torch.cat(
                [X_tensor, torch.tensor(cluster_probs, dtype=torch.float32)], dim=1
            )
            input_dim = X_tensor.shape[1]

        self._model = MDN(
            input_dim=input_dim,
            output_dim=output_dim,
            num_mixtures=self._num_mixtures,
            hidden_layers=self._hidden_layers,
            hidden_activation_fn_name=self._hidden_activation_fn_name,
        )
        self._model.to(self._device)

    def _get_optimizer_fn(self, name: str):
        optimizers = {
            OptimizerFunctionEnum.SGD: torch.optim.SGD,
            OptimizerFunctionEnum.ADAM: torch.optim.Adam,
            OptimizerFunctionEnum.RMSPROP: torch.optim.RMSprop,
            OptimizerFunctionEnum.GRADIENT_DESCENT: torch.optim.SGD,
        }

        optimizer_class = optimizers.get(name)
        if optimizer_class is None:
            raise ValueError(f"Unknown optimizer: {name}")
        return optimizer_class

    def _prepare_dataloaders(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        batch_size: int = 32,
        test_size: float = 0.2,
    ) -> tuple[DataLoader, DataLoader]:
        """Splits data and creates PyTorch DataLoaders for batch training."""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def fit(
        self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64], **kwargs
    ) -> None:
        """Fits the MDN model using batched data and early stopping."""
        super().fit(X, y)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(y, dtype=torch.float32)

        # Unpack additional keyword arguments
        epochs = kwargs.get("epochs", 100)
        batch_size = kwargs.get("batch_size", 32)

        self._prepare_model(X_tensor, Y_tensor)  # This helper function is not changed
        train_loader, val_loader = self._prepare_dataloaders(X, y, batch_size)

        optimizer_fn = self._get_optimizer_fn(self._optimizer_fn_name)
        optimizer = optimizer_fn(self._model.parameters(), lr=self._learning_rate)

        best_loss = float("inf")
        patience_counter = 0

        self._training_history = {"epochs": [], "train_loss": [], "val_loss": []}

        epochs_range = (
            tqdm(range(epochs), unit="epoch") if self._verbose else range(epochs)
        )

        for epoch in epochs_range:
            # Training loop
            self._model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                pi, mu, sigma = self._model(batch_X)
                dist = self._model._get_distribution(
                    pi, mu, sigma, self._distribution_family
                )
                loss = -dist.log_prob(batch_y).mean()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation loop
            self._model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    pi, mu, sigma = self._model(batch_X)
                    dist = self._model._get_distribution(
                        pi, mu, sigma, self._distribution_family
                    )
                    loss = -dist.log_prob(batch_y).mean()
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            if self._verbose:
                epochs_range.set_postfix(
                    train_loss=f"{avg_train_loss:.4f}", val_loss=f"{avg_val_loss:.4f}"
                )

            # Early stopping check
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0
                self._best_model_state_dict = self._model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self._early_stopping_patience:
                    if self._verbose:
                        print(
                            f"Early stopping at epoch {epoch}. Best validation loss: {best_loss:.4f}"
                        )
                    if self._best_model_state_dict:
                        self._model.load_state_dict(self._best_model_state_dict)
                    break

            self._training_history["epochs"].append(epoch)
            self._training_history["train_loss"].append(float(avg_train_loss))
            self._training_history["val_loss"].append(float(avg_val_loss))

        self._model.eval()

    def predict(
        self, X: npt.NDArray[np.float64], n_samples: int = 10, mode: str = "samples"
    ) -> npt.NDArray[np.float64]:
        if self._model is None:
            raise RuntimeError("The model has not been fit yet. Call 'fit' first.")

        X_tensor = torch.tensor(X, dtype=torch.float32)

        if self._gmm_boost and self._clusterer is not None:
            cluster_probs = self._clusterer.predict_proba(X)
            X_tensor = torch.cat(
                [X_tensor, torch.tensor(cluster_probs, dtype=torch.float32)], dim=1
            )

        with torch.no_grad():
            pi, mu, sigma = self._model(X_tensor)
            dist = self._model._get_distribution(
                pi, mu, sigma, self._distribution_family
            )

            if n_samples < 1:
                raise ValueError("n_samples must be at least 1.")

            if n_samples == 1:
                return dist.mean().numpy()

            if n_samples > 1 and mode == "samples":
                samples = dist.sample((n_samples,))
                return samples.numpy()

            elif (
                mode == "map"
            ):  # returns the mean (μ) of the most probable mixture component
                k_star = torch.argmax(pi, dim=1)
                y_hat = mu[torch.arange(mu.size(0)), k_star, :]
                return y_hat.numpy()

            elif mode == "mean":
                weighted_mean = torch.sum(pi.unsqueeze(-1) * mu, dim=1)
                return weighted_mean.numpy()

            elif mode == "median":
                weighted_median = torch.median(pi.unsqueeze(-1) * mu, dim=1)
                return weighted_median.numpy()
            else:
                raise ValueError(f"Unknown mode '{mode}'")

    def get_mixture_parameters(self, X: npt.NDArray[np.float64]):
        """
        Get the mixture parameters from the model.
        """
        if self._model is None:
            raise RuntimeError("The model has not been fit yet. Call 'fit' first.")

        X_tensor = torch.tensor(X, dtype=torch.float32)

        if self._gmm_boost and self._clusterer is not None:
            cluster_probs = self._clusterer.predict_proba(X)
            X_tensor = torch.cat(
                [X_tensor, torch.tensor(cluster_probs, dtype=torch.float32)], dim=1
            )

        with torch.no_grad():
            pi, mu, sigma = self._model(X_tensor)
            return pi.numpy(), mu.numpy(), sigma.numpy()


# ----------------------------------------------------------------------
# Helper Functions for Visualization and Data Handling
# ----------------------------------------------------------------------


def plot_predict_dist(
    model: MDNEstimator,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    non_linear: bool = False,
):
    """
    Plots the conditional mixture distributions using Plotly Express.

    Args:
        model: The trained MDNEstimator model.
        X: The input data (e.g., objectives).
        y: The output data (e.g., decisions).
        non_linear: Whether to use UMAP (True) or PCA (False) for dimensionality reduction.
    """
    pi, mu, sigma = model.get_mixture_parameters(X)
    X_reduced = _reduce_dim(X, non_linear)

    # Prepare data for plotting
    df_mixtures = pd.DataFrame(
        {
            "x": np.repeat(X_reduced.squeeze(), mu.shape[1]),
            "y": mu.reshape(-1, 1).squeeze(),
            "sigma": sigma.reshape(-1, 1).squeeze(),
            "pi": pi.reshape(-1, 1).squeeze(),
            "Mixture": np.tile(np.arange(mu.shape[1]), mu.shape[0]),
        }
    )

    # Create the scatter plot for mixture components
    fig = px.scatter(
        df_mixtures,
        x="x",
        y="y",
        color="Mixture",
        size="pi",  # Use pi for size
        hover_data={"sigma": True},
        title="Conditional Mixture Distributions",
        labels={"x": "Reduced Input Dimension", "y": "Output"},
    )

    # Add the true data points as a separate trace
    df_true = pd.DataFrame({"x": X_reduced.squeeze(), "y": y.squeeze()})
    fig.add_scatter(
        x=df_true["x"],
        y=df_true["y"],
        mode="markers",
        marker={"color": "black", "size": 5, "opacity": 0.2},
        name="True Data",
    )

    fig.show()


def plot_samples_vs_true(
    model: MDNEstimator,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    non_linear: bool = False,
):
    """
    Plots generated samples against true data using Plotly Express.

    Args:
        model: The trained MDNEstimator model.
        X: The input data (e.g., objectives).
        y: The output data (e.g., decisions).
        non_linear: Whether to use UMAP (True) or PCA (False) for dimensionality reduction.
    """
    samples = model.predict(X, mode="samples", n_samples=1)
    X_reduced = _reduce_dim(X, non_linear)

    df = pd.DataFrame(
        {
            "x": np.concatenate([X_reduced.squeeze(), X_reduced.squeeze()]),
            "y": np.concatenate([samples.squeeze(), y.squeeze()]),
            "Type": ["Generated Sample"] * len(samples) + ["True Data"] * len(y),
        }
    )

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="Type",
        title="Generated Samples vs True Data",
        labels={"x": "Reduced Input Dimension", "y": "Output"},
        opacity=0.4,
    )
    fig.show()


def _reduce_dim(
    X: npt.NDArray[np.float64], non_linear: bool
) -> npt.NDArray[np.float64]:
    if X.shape[1] > 1:
        if non_linear:
            reducer = UMAP(n_components=1)
        else:
            reducer = PCA(n_components=1)
        X_reduced = reducer.fit_transform(X)
    else:
        X_reduced = X
    return X_reduced
