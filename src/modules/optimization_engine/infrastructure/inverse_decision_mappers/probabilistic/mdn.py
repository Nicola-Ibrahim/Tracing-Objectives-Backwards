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
from umap import UMAP

from ....domain.model_management.interfaces.base_inverse_decision_mapper import (
    ProbabilisticInverseDecisionMapper,
)


class ActivationFunction(Enum):
    """An Enum for available activation function names."""

    RELU = "relu"
    TANH = "tanh"
    LEAKY_RELU = "leaky_relu"
    SIGMOID = "sigmoid"
    NONE = "none"


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
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_mixtures: int = 5,
        hidden_layers: list[int] = [64],
        hidden_activation: ActivationFunction = ActivationFunction.RELU,
        final_activation: ActivationFunction = ActivationFunction.RELU,
    ):
        """
        Initialize the Mixture Density Network (MDN) module.

        Args:
            input_dim (int): The dimensionality of the input space.
            output_dim (int): The dimensionality of the output space.
            num_mixtures (int): The number of mixture components.
            hidden_layers (list[int]): The sizes of the hidden layers.
            hidden_activation (ActivationFunction): The activation function for hidden layers.
            final_activation (ActivationFunction): The activation function for the output layer.
        """
        super().__init__()

        self.num_mixtures = num_mixtures
        self.output_dim = output_dim
        self.hidden_activation = self._get_activation(hidden_activation)
        self.final_activation = self._get_activation(final_activation)

        # Build the main body of the network
        layers = []
        in_size = input_dim
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(self.hidden_activation)
            in_size = hidden_size

        if layers:
            layers.pop()

        self.hidden_stack = nn.Sequential(*layers)

        # Output layers for mixture parameters
        final_hidden_size = hidden_layers[-1] if hidden_layers else input_dim
        self.fc_pi = nn.Linear(final_hidden_size, num_mixtures)
        self.fc_mu = nn.Linear(final_hidden_size, num_mixtures * output_dim)
        self.fc_sigma = nn.Linear(final_hidden_size, num_mixtures * output_dim)

    def _get_activation(self, activation_fn: ActivationFunction):
        """
        Get the activation function based on the specified activation function name.
        """

        activation_fns = {
            ActivationFunction.RELU: nn.ReLU(),
            ActivationFunction.TANH: nn.Tanh(),
            ActivationFunction.LEAKY_RELU: nn.LeakyReLU(),
            ActivationFunction.SIGMOID: nn.Sigmoid(),
            ActivationFunction.NONE: nn.Identity(),
        }

        try:
            return activation_fns[activation_fn]
        except KeyError:
            # This case should not be reachable if ActivationFunction is used
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
            # This case should not be reachable if DistributionFamily is used
            raise ValueError(
                f"Unsupported distribution family: {distribution_family.value}"
            )

    def forward(self, x: torch.Tensor):
        h = self.hidden_stack(x)
        h = self.final_activation(h)
        pi = F.softmax(self.fc_pi(h), dim=1)
        mu = self.fc_mu(h).view(-1, self.num_mixtures, self.output_dim)
        sigma = torch.exp(self.fc_sigma(h)).view(-1, self.num_mixtures, self.output_dim)
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
        hidden_activation_fn: ActivationFunction = ActivationFunction.RELU,
        final_activation_fn: ActivationFunction = ActivationFunction.RELU,
        optimizer_fn: OptimizerFunction = OptimizerFunction.ADAM,
    ):
        """
        Initialize the MDNInverseDecisionMapper.

        Args:
            num_mixtures: The number of mixtures for the MDN.
            epochs: The number of training epochs.
            learning_rate: The learning rate for the optimizer.
            early_stopping_patience: The patience for early stopping.
            distribution_family: The distribution family to use.
            gmm_boost: Whether to use GMM boosting.
            hidden_layers: The hidden layers for the MDN.
            hidden_activation_fn: The activation function for the hidden layers.
            final_activation_fn: The activation function for the final layer.
            optimizer_fn: The optimizer function to use.
        """
        super().__init__()
        self._num_mixtures = num_mixtures
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._early_stopping_patience = early_stopping_patience
        self._distribution_family = distribution_family
        self._gmm_boost = gmm_boost
        self._hidden_layers = hidden_layers
        self._hidden_activation_fn = hidden_activation_fn
        self._final_activation_fn = final_activation_fn
        self._optimizer_fn = optimizer_fn
        self._model: MDN | None = None
        self._clusterer = None
        self._best_model_state_dict = None

    def _determine_num_mixtures(self, X_y: npt.NDArray[np.float64]) -> int:
        lowest_bic = np.infty
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

        # Pass the Enum members directly to MDN's constructor
        self._model = MDN(
            input_dim=input_dim,
            output_dim=output_dim,
            num_mixtures=self._num_mixtures,
            hidden_layers=self._hidden_layers,
            hidden_activation=self._hidden_activation_fn,
            final_activation=self._final_activation_fn,
        )
        return X_tensor, Y_tensor

    def _get_optimizer_fn(self):
        optimizers = {
            OptimizerFunction.SGD: torch.optim.SGD,
            OptimizerFunction.ADAM: torch.optim.Adam,
            OptimizerFunction.RMSPROP: torch.optim.RMSprop,
            OptimizerFunction.GRADIENT_DESCENT: torch.optim.SGD,
        }

        # Use self._optimizer_fn directly from the class instance
        optimizer_class = optimizers.get(self._optimizer_fn)
        if optimizer_class is None:
            raise ValueError(f"Unknown optimizer: {self._optimizer_fn}")
        return optimizer_class

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> None:
        super().fit(X, y)

        X_tensor, Y_tensor = self._prepare_data_and_model(X, y)

        optimizer_fn = self._get_optimizer_fn()
        optimizer = optimizer_fn(self._model.parameters(), lr=self._learning_rate)
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self._epochs):
            self._model.train()
            optimizer.zero_grad()
            pi, mu, sigma = self._model(X_tensor)
            dist = self._model._get_distribution(
                pi, mu, sigma, self._distribution_family
            )
            loss = -dist.log_prob(Y_tensor).mean()
            loss.backward()
            optimizer.step()

            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
                self._best_model_state_dict = self._model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self._early_stopping_patience:
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
