import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ...domain.interfaces.base_estimator import ProbabilisticEstimator


class _MDNNetwork(nn.Module):
    """
    Core Mixture Density Network architecture.
    Maps input features to parameters of a Gaussian Mixture Model (pi, mu, sigma).
    """

    def __init__(self, input_dim: int, output_dim: int, n_hidden: int, n_mixtures: int):
        super().__init__()
        self.z_h = nn.Sequential(
            nn.Linear(input_dim, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
        )
        self.z_pi = nn.Linear(n_hidden, n_mixtures)
        self.z_mu = nn.Linear(n_hidden, n_mixtures * output_dim)
        self.z_sigma = nn.Linear(n_hidden, n_mixtures * output_dim)

        self.output_dim = output_dim
        self.n_mixtures = n_mixtures

    def forward(self, x):
        h = self.z_h(x)
        pi = torch.softmax(self.z_pi(h), dim=1)
        mu = self.z_mu(h).view(-1, self.n_mixtures, self.output_dim)
        # Using softplus for sigma to ensure positivity
        sigma = torch.nn.functional.softplus(self.z_sigma(h)).view(
            -1, self.n_mixtures, self.output_dim
        )
        return pi, mu, sigma


def mdn_loss(pi, mu, sigma, y):
    """
    Negative Log-Likelihood loss for MDN.
    """
    # Expand y to match mu/sigma dimensions: (batch, n_mixtures, output_dim)
    y = y.unsqueeze(1).expand_as(mu)

    # Calculate probability density of y under each mixture component
    # Using torch.distributions.Normal for numerical stability
    m = torch.distributions.Normal(loc=mu, scale=sigma + 1e-6)
    log_prob = m.log_prob(y)  # (batch, n_mixtures, output_dim)

    # Sum log probabilities across output dimensions (assuming independence within component)
    log_prob = torch.sum(log_prob, dim=2)  # (batch, n_mixtures)

    # Weighted sum of probabilities across mixtures
    # Use log-sum-exp for numerical stability
    # log(sum(pi * exp(log_prob))) = log(sum(exp(log(pi) + log_prob)))
    combined_log_prob = torch.log(pi + 1e-8) + log_prob
    nll = -torch.logsumexp(combined_log_prob, dim=1)

    return torch.mean(nll)


class MDNEstimator(ProbabilisticEstimator):
    """
    Mixture Density Network (MDN) Estimator.
    Learns p(decisions | objectives) as a mixture of Gaussians.
    """

    def __init__(
        self,
        n_hidden: int = 64,
        n_mixtures: int = 5,
        epochs: int = 200,
        lr: float = 0.001,
        batch_size: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_hidden = n_hidden
        self.n_mixtures = n_mixtures
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.network = None

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64], **kwargs):
        """
        Trains the MDN on the provided data.
        X: Objectives (inputs to MDN)
        y: Decisions (targets of MDN)
        """
        super().fit(X, y)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.network = _MDNNetwork(
            input_dim=X.shape[1],
            output_dim=y.shape[1],
            n_hidden=self.n_hidden,
            n_mixtures=self.n_mixtures,
        )
        optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        self.network.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                pi, mu, sigma = self.network(batch_X)
                loss = mdn_loss(pi, mu, sigma, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            self._training_history.log(epoch, train_loss=avg_loss)

    def sample(
        self,
        X: npt.NDArray[np.float64],
        n_samples: int = 1,
        seed: int = 42,
        **kwargs,
    ) -> npt.NDArray[np.float64]:
        """
        Draws samples from the learned p(y|X).
        """
        if self.network is None:
            raise RuntimeError("MDNEstimator must be fitted before sampling.")

        torch.manual_seed(seed)
        X_tensor = torch.tensor(X, dtype=torch.float32)

        self.network.eval()
        with torch.no_grad():
            pi, mu, sigma = self.network(X_tensor)

            # pi: (N, n_mixtures)
            # mu: (N, n_mixtures, output_dim)
            # sigma: (N, n_mixtures, output_dim)

            N = X.shape[0]

            # Choose mixture components for each sample
            # We want to draw n_samples for each input point in X
            # Multinomial expects (N, n_mixtures) and draws (N, n_samples)
            comp_indices = torch.multinomial(
                pi, n_samples, replacement=True
            )  # (N, n_samples)

            # Gather mu and sigma for chosen components
            # This requires manual indexing or reshaping
            all_samples = []
            for i in range(N):
                # For input i, we have n_samples mixture choices
                indices = comp_indices[i]  # (n_samples,)
                m = mu[i, indices]  # (n_samples, output_dim)
                s = sigma[i, indices]  # (n_samples, output_dim)

                # Sample from chosen Gaussians
                dist = torch.distributions.Normal(m, s + 1e-6)
                samples = dist.sample()  # (n_samples, output_dim)
                all_samples.append(samples.numpy())

            return np.stack(all_samples)  # (N, n_samples, output_dim)

    def get_log_likelihood(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculates log-likelihood for given pairs.
        """
        if self.network is None:
            return np.zeros(X.shape[0])

        self.network.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        with torch.no_grad():
            pi, mu, sigma = self.network(X_tensor)
            # Expanded y
            y_exp = y_tensor.unsqueeze(1).expand_as(mu)
            m = torch.distributions.Normal(loc=mu, scale=sigma + 1e-6)
            log_prob = m.log_prob(y_exp)
            log_prob = torch.sum(log_prob, dim=2)
            combined_log_prob = torch.log(pi + 1e-8) + log_prob
            ll = torch.logsumexp(combined_log_prob, dim=1)
            return ll.numpy()

    def to_checkpoint(self) -> dict:
        """MDN-specific serialization."""
        return {
            "type": self.type,
            "params": self._collect_init_params_from_instance(),
            "loss_history": self.get_loss_history(),
        }

    @classmethod
    def from_checkpoint(cls, parameters: dict) -> "MDNEstimator":
        params = parameters.get("params", {})
        return cls(**params)
