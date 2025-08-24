from enum import Enum

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn
import torch.nn.functional as F
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from torch.distributions import (
    Categorical,
    Gamma,
    Independent,
    Laplace,
    LogNormal,
    MixtureSameFamily,
    Normal,
)
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from umap import UMAP

from ....domain.model_management.interfaces.base_inverse_decision_mapper import (
    ProbabilisticInverseDecisionMapper,
)


class ActivationFunction(Enum):
    """An Enum for available activation function names."""

    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SOFTPLUS = "softplus"
    IDENTITY = "identity"


class DistributionFamily(Enum):
    """An Enum for available distribution families."""

    NORMAL = "normal"
    LAPLACE = "laplace"
    LOGNORMAL = "lognormal"


class OptimizerFunction(Enum):
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
        hidden_activation_fn_name: ActivationFunction = ActivationFunction.RELU,
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

        # Means (μ) → shape (num_mixtures, output_dim)
        self.fc_mu = nn.Linear(final_hidden_size, num_mixtures * output_dim)

        # Standard deviations (σ) → positive via softplus/exp
        self.fc_sigma = nn.Linear(final_hidden_size, num_mixtures * output_dim)

    # ------- Utility Functions -------
    def _get_activation(self, activation_fn: ActivationFunction):
        """
        Get the activation function based on the specified activation function name.
        """
        activation_fns = {
            ActivationFunction.RELU: nn.ReLU(),
            ActivationFunction.TANH: nn.Tanh(),
            ActivationFunction.SIGMOID: nn.Sigmoid(),
            ActivationFunction.SOFTPLUS: nn.Softplus(),
            ActivationFunction.IDENTITY: nn.Identity(),
        }

        try:
            return activation_fns[activation_fn]
        except KeyError:
            raise ValueError(f"Unsupported activation function: {activation_fn.value}")

    @staticmethod
    def _get_distribution(
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        distribution_family: DistributionFamily,
    ):
        """
        Get the mixture distribution based on the specified distribution family.
        """
        distribution_families = {
            DistributionFamily.NORMAL: MixtureSameFamily(
                Categorical(pi), Independent(Normal(mu, sigma), 1)
            ),
            DistributionFamily.LAPLACE: MixtureSameFamily(
                Categorical(pi), Independent(Laplace(mu, sigma), 1)
            ),
            DistributionFamily.LOGNORMAL: MixtureSameFamily(
                Categorical(pi), Independent(LogNormal(mu, sigma), 1)
            ),
        }

        try:
            return distribution_families[distribution_family]
        except KeyError:
            raise ValueError(
                f"Unsupported distribution family: {distribution_family.value}"
            )

    def summary(self, input_size=(1,)):
        """
        Prints a summary of the architecture with tqdm visualization.
        """
        x = torch.zeros(1, *input_size)
        print("\nMixture Density Network (MDN) Summary\n" + "=" * 50)

        layers = []
        layers.append(("Input", x.shape))

        # Go through hidden stack
        tmp_x = x
        for i, layer in enumerate(
            tqdm(self.hidden_stack, desc="Building Summary", unit="layer")
        ):
            tmp_x = layer(tmp_x)
            layers.append((layer.__class__.__name__, tmp_x.shape))

        # Heads
        pi = torch.softmax(self.fc_pi(tmp_x), dim=-1)
        mu = self.fc_mu(tmp_x).view(-1, self.num_mixtures, self.output_dim)
        sigma = torch.exp(self.fc_sigma(tmp_x)).view(
            -1, self.num_mixtures, self.output_dim
        )

        layers.append(("π (mixture weights)", pi.shape))
        layers.append(("μ (means)", mu.shape))
        layers.append(("σ (stddev)", sigma.shape))

        # Print results
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
        pi = F.softmax(pi, dim=-1)

        # Calculate means (mu) and reshape to (batch_size, num_mixtures, output_dim)
        mu = self.fc_mu(h)  # [batch_size, num_mixtures * output_dim]
        mu = mu.view(-1, self.num_mixtures, self.output_dim)

        # Calculate standard deviations (sigma) by exponentiating the output
        # and reshape to (batch_size, num_mixtures, output_dim)
        sigma = self.fc_sigma(h)  # [batch_size, num_mixtures * output_dim]
        sigma = F.softplus(sigma) + 1e-6
        sigma = sigma.view(-1, self.num_mixtures, self.output_dim)

        return pi, mu, sigma


class MDNInverseDecisionMapper(ProbabilisticInverseDecisionMapper):
    """
    An inverse mapper that uses a Mixture Density Network (MDN) to model the
    inverse relationship from objectives to decisions.
    """

    def __init__(
        self,
        num_mixtures: int = -1,
        epochs: int = 500,
        learning_rate: float = 1e-3,
        early_stopping_patience: int = 10,
        distribution_family: DistributionFamily = DistributionFamily.NORMAL,
        gmm_boost: bool = False,
        hidden_layers: list[int] = [64],
        hidden_activation_fn_name: ActivationFunction = ActivationFunction.RELU,
        optimizer_fn_name: OptimizerFunction = OptimizerFunction.ADAM,
        verbose: bool = False,
    ):
        """
        Initialize the MDNInverseDecisionMapper.
        """
        super().__init__()
        self._num_mixtures = num_mixtures
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._early_stopping_patience = early_stopping_patience
        self._distribution_family = distribution_family
        self._gmm_boost = gmm_boost
        self._hidden_layers = hidden_layers
        self._hidden_activation_fn_name = hidden_activation_fn_name
        self._optimizer_fn_name = optimizer_fn_name
        self._verbose = verbose
        self._model: MDN | None = None
        self._clusterer = None
        self._best_model_state_dict = None

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

    def _prepare_data_and_model(
        self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(y, dtype=torch.float32)

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
        return X_tensor, Y_tensor

    def _get_optimizer_fn(self, name: str):
        optimizers = {
            OptimizerFunction.SGD: torch.optim.SGD,
            OptimizerFunction.ADAM: torch.optim.Adam(),
            OptimizerFunction.RMSPROP: torch.optim.RMSprop,
            OptimizerFunction.GRADIENT_DESCENT: torch.optim.SGD,
        }

        optimizer_class = optimizers.get(name)
        if optimizer_class is None:
            raise ValueError(f"Unknown optimizer: {name}")
        return optimizer_class

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> None:
        super().fit(X, y)

        X_tensor, Y_tensor = self._prepare_data_and_model(X, y)
        optimizer_fn = self._get_optimizer_fn(self._optimizer_fn_name)
        optimizer = optimizer_fn(self._model.parameters(), lr=self._learning_rate)

        best_loss = float("inf")
        patience_counter = 0

        # Conditional tqdm based on the verbose flag
        epochs_range = range(self._epochs)
        if self._verbose:
            epochs_range = tqdm(epochs_range, unit="epoch")

        for epoch in epochs_range:
            self._model.train()
            optimizer.zero_grad()
            pi, mu, sigma = self._model(X_tensor)
            dist = self._model._get_distribution(
                pi, mu, sigma, self._distribution_family
            )
            loss = -dist.log_prob(Y_tensor).mean()
            loss.backward()

            optimizer.step()

            # Update the progress bar if verbose is enabled
            if self._verbose:
                epochs_range.set_postfix(loss=f"{loss.item():.4f}")

            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
                self._best_model_state_dict = self._model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self._early_stopping_patience:
                    if self._verbose:
                        print(
                            f"Early stopping at epoch {epoch}. Best loss: {best_loss:.4f}"
                        )
                    if self._best_model_state_dict:
                        self._model.load_state_dict(self._best_model_state_dict)
                    break
        self._model.eval()

    def predict(
        self,
        X: npt.NDArray[np.float64],
        mode: str = "samples",
        n_samples: int = 10,
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

            if mode == "samples":
                samples = dist.sample((n_samples,))
                return samples.numpy()
            elif mode == "map":
                k_star = torch.argmax(pi, dim=1)
                y_hat = mu[torch.arange(mu.size(0)), k_star, :]
                return y_hat.numpy()
            elif mode == "mean":
                weighted_mean = torch.sum(pi.unsqueeze(-1) * mu, dim=1)
                return weighted_mean.numpy()
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
    model: MDNInverseDecisionMapper,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    non_linear: bool = False,
):
    """
    Plots the conditional mixture distributions using Plotly Express.

    Args:
        model: The trained MDNInverseDecisionMapper model.
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
    model: MDNInverseDecisionMapper,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    non_linear: bool = False,
):
    """
    Plots generated samples against true data using Plotly Express.

    Args:
        model: The trained MDNInverseDecisionMapper model.
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
