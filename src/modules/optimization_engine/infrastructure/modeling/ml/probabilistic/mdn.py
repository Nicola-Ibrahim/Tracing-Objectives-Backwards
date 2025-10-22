from enum import Enum
from typing import Sequence

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
    Independent,
    Laplace,
    LogNormal,
    MixtureSameFamily,
    Normal,
)
from torch.utils.data import DataLoader, TensorDataset
from umap import UMAP

from .....domain.modeling.enums.estimator_type import EstimatorTypeEnum
from .....domain.modeling.interfaces.base_estimator import (
    ProbabilisticEstimator,
)

# ======================================================================
# Data containers & Enums
# ======================================================================


class ActivationFunctionEnum(Enum):
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SOFTPLUS = "softplus"
    IDENTITY = "identity"


class DistributionFamilyEnum(Enum):
    NORMAL = "normal"
    LAPLACE = "laplace"
    LOGNORMAL = "lognormal"  # interpret mu, sigma as log-space params


class OptimizerFunctionEnum(Enum):
    SGD = "sgd"
    ADAM = "adam"
    RMSPROP = "rmsprop"
    GRADIENT_DESCENT = "gradient_descent"  # alias to SGD


# ======================================================================
# MDN module
# ======================================================================


class MDN(nn.Module):
    """
    A Mixture Density Network (MDN) with configurable hidden stack and heads.

    Heads:
      - π-head:   Linear -> Softmax over K
      - μ-head:   Linear -> reshape (K, out_dim)
      - σ-head:   Linear -> Softplus + eps -> reshape (K, out_dim)  (σ is SCALE)

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


    NOTE: For LOGNORMAL, μ and σ are log-space parameters (location & scale).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_mixtures: int = 5,
        hidden_layers: list[int] = [64, 32],
        hidden_activation_fn_name: ActivationFunctionEnum = ActivationFunctionEnum.RELU,
        pi_bias_init: Sequence[float] = None,
        mu_bias_init: Sequence[float] = None,
    ):
        super().__init__()

        self.num_mixtures = int(num_mixtures)
        self.output_dim = int(output_dim)

        # Activation (module instance)
        self.hidden_activation = self._get_activation(hidden_activation_fn_name)
        self.hidden_activation_name = hidden_activation_fn_name

        # ----- Build hidden stack explicitly (no trailing activation) -----
        layers: list[nn.Module] = []
        in_size = input_dim
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(in_size, hidden_size))
            if i < len(hidden_layers) - 1:
                layers.append(self.hidden_activation)
            in_size = hidden_size
        self.hidden_stack = nn.Sequential(*layers)
        final_hidden_size = hidden_layers[-1] if hidden_layers else input_dim

        # ----- Heads -----
        self.fc_pi = nn.Linear(final_hidden_size, num_mixtures)
        nn.init.normal_(self.fc_pi.weight)

        self.fc_mu = nn.Linear(final_hidden_size, num_mixtures * self.output_dim)
        self.fc_sigma = nn.Linear(final_hidden_size, num_mixtures * self.output_dim)
        nn.init.normal_(self.fc_sigma.weight)

        # Optional bias initialisation with size checks
        if pi_bias_init is not None:
            b = torch.tensor(pi_bias_init, dtype=self.fc_pi.bias.dtype)
            if b.numel() != self.fc_pi.bias.numel():
                raise ValueError("pi_bias_init size mismatch.")
            with torch.no_grad():
                self.fc_pi.bias.copy_(b)

        if mu_bias_init is not None:
            b = torch.tensor(mu_bias_init, dtype=self.fc_mu.bias.dtype)
            if b.numel() != self.fc_mu.bias.numel():
                raise ValueError("mu_bias_init size mismatch.")
            with torch.no_grad():
                self.fc_mu.bias.copy_(b)

    # ------- Utility Functions -------
    def _get_activation(self, activation_fn: ActivationFunctionEnum) -> nn.Module:
        mapping = {
            ActivationFunctionEnum.RELU: nn.ReLU(),
            ActivationFunctionEnum.TANH: nn.Tanh(),
            ActivationFunctionEnum.SIGMOID: nn.Sigmoid(),
            ActivationFunctionEnum.SOFTPLUS: nn.Softplus(),
            ActivationFunctionEnum.IDENTITY: nn.Identity(),
        }
        return mapping.get(activation_fn, nn.Identity())

    def _get_distribution(
        self,
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        distribution_family: DistributionFamilyEnum,
    ) -> MixtureSameFamily:
        """
        Builds a MixtureSameFamily with component distribution:
          - Normal(mu, sigma)     if NORMAL
          - Laplace(mu, sigma)    if LAPLACE
          - LogNormal(mu, sigma)  if LOGNORMAL  (mu, sigma are log-space)
        """

        mapping = {
            DistributionFamilyEnum.NORMAL: Normal(mu, sigma),
            DistributionFamilyEnum.LAPLACE: Laplace(mu, sigma),
            DistributionFamilyEnum.LOGNORMAL: LogNormal(mu, sigma),
        }

        comp = mapping[distribution_family]

        return MixtureSameFamily(Categorical(probs=pi), Independent(comp, 1))

    def summary(self, input_size=(1,)):
        x = torch.zeros(1, *input_size)
        print("\nMixture Density Network (MDN) Summary\n" + "=" * 50)
        layers = [("Input", x.shape)]
        tmp_x = x
        for i, layer in enumerate(self.hidden_stack):
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
            F.softplus(sigma) + 1e-2
        )  # Enforce positive stddev so the Gaussian is well-defined and gradients behave predictably
        sigma = sigma.view(-1, self.num_mixtures, self.output_dim)
        return pi, mu, sigma


# ======================================================================
# Estimator
# ======================================================================


class MDNEstimator(ProbabilisticEstimator):
    """
    Probabilistic inverse mapper using an MDN.

    Public API:
      - predict(X, ...) -> one stochastic draw per x: (n, out_dim)
      - sample(X, n_samples, ...) -> many draws: (n, n_samples, out_dim)
      - infer_mean / infer_median / infer_map
      - get_mixture_parameters(X) -> (pi, mu, sigma) as numpy arrays
    """

    def __init__(
        self,
        num_mixtures: int = -1,
        learning_rate: float = 1e-4,
        distribution_family: DistributionFamilyEnum = DistributionFamilyEnum.NORMAL,
        gmm_boost: bool = False,
        hidden_layers: list[int] = [256, 128, 128],
        hidden_activation_fn_name: ActivationFunctionEnum = ActivationFunctionEnum.RELU,
        optimizer_fn_name: OptimizerFunctionEnum = OptimizerFunctionEnum.ADAM,
        verbose: bool = False,
        epochs: int = 100,
        batch_size: int = 32,
        val_size: float = 0.2,
        weight_decay: float = 0.0,
        clip_grad_norm: float | None = None,
        seed: int = 44,
    ):
        super().__init__()
        self._num_mixtures = num_mixtures
        self._learning_rate = learning_rate
        self._distribution_family = distribution_family
        self._gmm_boost = gmm_boost
        self._hidden_layers = hidden_layers
        self._hidden_activation_fn_name = hidden_activation_fn_name
        self._optimizer_fn_name = optimizer_fn_name
        self._verbose = verbose

        self._model: MDN | None = None
        self._clusterer: GaussianMixture = None
        self._best_model_state_dict: dict = None

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epochs = epochs
        self.batch_size = batch_size
        self.val_size = val_size
        self.weight_decay = weight_decay
        self.clip_grad_norm = clip_grad_norm
        self.seed = seed

    @property
    def type(self) -> str:
        return getattr(EstimatorTypeEnum, "MDN", EstimatorTypeEnum.MDN).value

    # ----------------------- Fitting -----------------------

    def _determine_num_mixtures(self, X_y: npt.NDArray[np.float64]) -> int:
        lowest_bic = np.inf
        best_gmm: GaussianMixture = None
        for n_components in range(1, 7):
            gmm = GaussianMixture(
                n_components=n_components, covariance_type="full", max_iter=10000
            )
            gmm.fit(X_y)
            bic = gmm.bic(X_y)
            if bic < lowest_bic:
                lowest_bic = bic
                best_gmm = gmm
        if best_gmm is None:
            raise RuntimeError("Failed to determine number of mixtures via BIC.")
        self._clusterer = best_gmm
        return best_gmm.n_components

    def _prepare_model(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
        input_dim = X.shape[1]
        output_dim = y.shape[1]

        if self._num_mixtures == -1:
            self._num_mixtures = self._determine_num_mixtures(np.hstack((X, y)))

        if self._gmm_boost:
            if self._clusterer is None:
                raise RuntimeError("GMM boost enabled but _clusterer is None.")
            cluster_probs = self._clusterer.predict_proba(X)
            input_dim = input_dim + cluster_probs.shape[1]

        self._model = MDN(
            input_dim=input_dim,
            output_dim=output_dim,
            num_mixtures=self._num_mixtures,
            hidden_layers=self._hidden_layers,
            hidden_activation_fn_name=self._hidden_activation_fn_name,
        ).to(self._device)

    def _get_optimizer_fn(self, name: OptimizerFunctionEnum):
        mapping = {
            OptimizerFunctionEnum.SGD: torch.optim.SGD,
            OptimizerFunctionEnum.ADAM: torch.optim.Adam,
            OptimizerFunctionEnum.RMSPROP: torch.optim.RMSprop,
            OptimizerFunctionEnum.GRADIENT_DESCENT: torch.optim.SGD,
        }
        return mapping[name]

    def _prepare_dataloaders(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> tuple[DataLoader, DataLoader]:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_size, random_state=42
        )
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def _maybe_fail_on_lognormal(self, y: npt.NDArray[np.float64]) -> None:
        if self._distribution_family == DistributionFamilyEnum.LOGNORMAL and np.any(
            y <= 0.0
        ):
            raise ValueError(
                "LogNormal selected but targets contain non-positive values. "
                "Either transform targets to be positive or use a different distribution."
            )

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> None:
        """Fit the MDN using the configured training hyperparameters."""
        super().fit(X, y)

        # Optional reproducibility
        seed = self.seed
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

        self._maybe_fail_on_lognormal(y)
        self._prepare_model(X, y)
        train_loader, val_loader = self._prepare_dataloaders(X, y)

        optimizer_fn = self._get_optimizer_fn(self._optimizer_fn_name)
        optimizer = optimizer_fn(
            self._model.parameters(),
            lr=self._learning_rate,
            weight_decay=self.weight_decay,
        )

        best_loss = float("inf")
        self._best_model_state_dict = None

        # Training loop
        for epoch in range(self.epochs):
            self._model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self._device)
                batch_y = batch_y.to(self._device)

                optimizer.zero_grad()
                pi, mu, sigma = self._model(batch_X)
                dist = self._model._get_distribution(
                    pi, mu, sigma, self._distribution_family
                )
                loss = -dist.log_prob(batch_y).mean()
                loss.backward()
                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(), self.clip_grad_norm
                    )
                optimizer.step()
                train_loss += loss.item()

            avg_train = train_loss / max(1, len(train_loader))

            # Validation
            self._model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self._device)
                    batch_y = batch_y.to(self._device)
                    pi, mu, sigma = self._model(batch_X)
                    dist = self._model._get_distribution(
                        pi, mu, sigma, self._distribution_family
                    )
                    val_loss += (-dist.log_prob(batch_y).mean()).item()

            avg_val = val_loss / max(1, len(val_loader))

            # Track best validation performance
            if avg_val < best_loss:
                best_loss = avg_val
                self._best_model_state_dict = self._model.state_dict()

            self._training_history.epochs.append(epoch)
            self._training_history.train_loss.append(float(avg_train))
            self._training_history.val_loss.append(float(avg_val))
        if self._best_model_state_dict is not None:
            self._model.load_state_dict(self._best_model_state_dict)

        self._model.eval()

    # ----------------------- Inference API -----------------------

    def sample(
        self,
        X: npt.NDArray[np.float64],
        n_samples: int = 1,
        seed: int = 42,
        temperature: float = 0.5,  # component spread temperature
        tau_pi: float = 0.5,  # mixture-weight (softmax) temperature
        max_outputs_per_chunk: int = 20_000,
    ) -> npt.NDArray[np.float64]:
        """
        Draw `n_samples` IID samples per input from p(x|y).

        Tempering:
        - `temperature` affects component dispersions as per family:
            Normal/LogNormal: std <- sqrt(T) * std; Laplace: scale <- T * scale.
        - `tau_pi` affects mixture selection via softmax temperature (higher => softer).

        Shapes:
        X: (n, in_dim)
        return: (n, out_dim)              if n_samples == 1
                (n, n_samples, out_dim)   if n_samples  > 1
        """
        self._ensure_fitted()
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")

        X_tensor = self._prepare_inputs(X)
        device = self._model_device()
        n = X_tensor.shape[0]

        # RNG
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        total_outputs = n * max(1, n_samples)

        def _sample_from(pi, mu, sigma, n_s):
            # apply temperatures
            if tau_pi != 1.0:
                pi = self._temper_pi(pi, tau_pi)
            if temperature != 1.0:
                sigma = self._apply_temperature(sigma, temperature)

            dist = self._model._get_distribution(
                pi, mu, sigma, self._distribution_family
            )
            if n_s == 1:
                y = dist.sample()  # (n, out)
                return y
            else:
                y = dist.sample((n_s,))  # (n_s, n, out)
                return y.permute(1, 0, 2)  # (n, n_s, out)

        # Fast path
        if total_outputs <= max_outputs_per_chunk:
            with torch.no_grad():
                pi, mu, sigma = self._model(X_tensor)
                out = _sample_from(pi, mu, sigma, n_samples)
            return out.cpu().numpy().astype(np.float64, copy=False)

        # Chunked path
        results = []
        chunk_batch = max(1, max_outputs_per_chunk // max(1, n_samples))
        for start in range(0, n, chunk_batch):
            end = min(n, start + chunk_batch)
            X_chunk = X_tensor[start:end]
            with torch.no_grad():
                pi_c, mu_c, sigma_c = self._model(X_chunk)
                s_chunk = _sample_from(pi_c, mu_c, sigma_c, n_samples)
                results.append(s_chunk.cpu())

        out = torch.cat(results, dim=0).numpy().astype(np.float64, copy=False)
        return out

    def infer_mean(
        self,
        X: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        **Monte-Carlo mean** per input: average of `n_samples` draws from p(x|y).

        Rationale:
        - Matches CVAE-style APIs and your `generate_point_predictions` idea.
        - Avoids relying on analytic mixture moments when models differ.

        Determinism:
        - Stochastic unless you fix `seed`. Larger `n_samples` → lower MC noise.

        Shapes:
        - X: (n, in_dim)
        - return: (n, out_dim)
        """
        s = self.sample(
            X,
            n_samples=256,
        )
        if s.ndim == 2:
            return s.astype(np.float64, copy=False)  # n_samples==1, trivial mean
        return s.mean(axis=1).astype(np.float64, copy=False)

    def infer_median(
        self,
        X: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        **Monte-Carlo median** per input (marginal, per output dimension).

        Shapes:
        - X: (n, in_dim)
        - return: (n, out_dim)
        """
        s = self.sample(
            X,
            n_samples=256,
        )
        if s.ndim == 2:
            return s.astype(np.float64, copy=False)
        return np.median(s, axis=1).astype(np.float64, copy=False)

    def infer_map(
        self,
        X: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        **Deterministic** mean of the MAP component per input:
        k* = argmax_k π_k(y); returns μ_{k*}(y).

        Shapes:
        - X: (n, in_dim)
        - return: (n, out_dim)
        """
        self._ensure_fitted()
        X_t = self._prepare_inputs(X)
        with torch.no_grad():
            pi, mu, sigma = self._model(X_t)  # (n,K,out)
            score = torch.log(pi) - torch.log(sigma).sum(dim=-1)  # (n,K)
            k_star = torch.argmax(score, dim=1)
            out = mu[torch.arange(mu.size(0), device=mu.device), k_star, :]
        return out.cpu().numpy().astype(np.float64, copy=False)

    def predict_topk(self, X: np.ndarray, K: int = 3, by: str = "score") -> np.ndarray:
        """
        Return top-K component means per input, shape (n, K, out_dim).
        by: "pi" or "score" (log pi - sum log sigma)
        """
        self._ensure_fitted()
        X_t = self._prepare_inputs(X)
        with torch.no_grad():
            pi, mu, sigma = self._model(X_t)  # (n,K,out)
            if by == "pi":
                key = pi
            else:
                key = torch.log(pi) - torch.log(sigma).sum(dim=-1)  # (n,K)
            topk = torch.topk(key, k=min(K, pi.shape[1]), dim=1).indices  # (n,K)
            n = X_t.size(0)
            out = []
            for i in range(n):
                out.append(mu[i, topk[i], :])
            out = torch.stack(out, dim=0)  # (n,K,out_dim)
        return out.cpu().numpy().astype(np.float64, copy=False)

    def get_mixture_parameters(
        self, X: npt.NDArray[np.float64]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (pi, mu, sigma) as numpy arrays."""
        self._ensure_fitted()
        X_tensor = self._prepare_inputs(X)
        with torch.no_grad():
            pi, mu, sigma = self._model(X_tensor)
        return pi.cpu().numpy(), mu.cpu().numpy(), sigma.cpu().numpy()

    # -------------------- internal helpers --------------------

    def _ensure_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("Estimator not fitted. Call 'fit' first.")

    def _model_device(self) -> torch.device:
        return next(self._model.parameters()).device

    def _prepare_inputs(self, X: npt.NDArray[np.float64]) -> torch.Tensor:
        """
        Convert to torch, optionally append GMM-boost features, and move to device.
        """
        device = self._model_device()
        X_t = torch.tensor(X, dtype=torch.float32, device=device)

        if self._gmm_boost and self._clusterer is not None:
            cluster_probs = self._clusterer.predict_proba(X)  # numpy
            cp_t = torch.tensor(cluster_probs, dtype=torch.float32, device=device)
            X_t = torch.cat([X_t, cp_t], dim=1)

        return X_t

    # --- add these helpers inside MDNEstimator ---------------------------------

    def _temper_pi(self, pi: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Softmax temperature on mixture weights already in probability space.
        tau > 1.0 => softer (higher entropy); tau < 1.0 => peakier.
        Implemented as: normalize(pi ** (1/tau)).
        """
        if tau == 1.0:
            return pi
        inv_tau = 1.0 / float(tau)
        # clamp to avoid 0**inv_tau -> NaNs and to keep gradients sane
        q = torch.clamp(pi, min=1e-12).pow(inv_tau)
        return q / q.sum(dim=-1, keepdim=True)

    def _apply_temperature(
        self, sigma: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        """
        Temper component scales according to the distribution family.

        NORMAL    : sigma_T = sqrt(T) * sigma        (variance ∝ T)
        LAPLACE   : b_T     = T * b                  (scale ∝ T)
        LOGNORMAL : sigma_T = sqrt(T) * sigma        (std in *log-space*)

        Final clamp protects against extreme values after scaling.
        """
        T = float(temperature)
        if T == 1.0:
            return sigma
        if self._distribution_family == DistributionFamilyEnum.LAPLACE:
            sigma_T = sigma * T
        else:
            # NORMAL and LOGNORMAL temper with sqrt(T) on std
            sigma_T = sigma * np.sqrt(T)
        return torch.clamp(sigma_T, min=1e-4)  # be conservative with a small floor


# ======================================================================
# Visualization & helpers
# ======================================================================


def plot_predict_dist(
    model: MDNEstimator,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    non_linear: bool = False,
):
    """
    Plot conditional mixture components (μ) against reduced X, with size ~ π.
    For multi-dimensional y, uses y[..., 0].
    """
    pi, mu, sigma = model.get_mixture_parameters(X)
    X_reduced = _reduce_dim(X, non_linear).reshape(-1)

    # select first output dimension for plotting if needed
    if mu.ndim == 3 and mu.shape[-1] > 1:
        mu_plot = mu[..., 0]
        sigma_plot = sigma[..., 0]
        y_plot = y[..., 0]
    else:
        mu_plot = mu.reshape(mu.shape[0], mu.shape[1])
        sigma_plot = sigma.reshape(sigma.shape[0], sigma.shape[1])
        y_plot = y.reshape(y.shape[0])

    df_mixtures = pd.DataFrame(
        {
            "x": np.repeat(X_reduced, mu_plot.shape[1]),
            "y": mu_plot.reshape(-1),
            "sigma": sigma_plot.reshape(-1),
            "pi": pi.reshape(-1),
            "Mixture": np.tile(np.arange(mu_plot.shape[1]), mu_plot.shape[0]),
        }
    )

    fig = px.scatter(
        df_mixtures,
        x="x",
        y="y",
        color="Mixture",
        size="pi",
        hover_data={"sigma": True},
        title="Conditional Mixture Distributions",
        labels={"x": "Reduced Input Dimension", "y": "Output (dim 0)"},
    )

    df_true = pd.DataFrame({"x": X_reduced, "y": y_plot})
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
    Compare one generated sample vs ground truth along reduced X.
    For multi-dimensional y, uses y[..., 0].
    """
    samples = model.sample(X, n_samples=1)  # (n, out)
    X_reduced = _reduce_dim(X, non_linear).reshape(-1)

    s = samples
    if s.ndim == 2 and s.shape[1] > 1:
        s_plot = s[:, 0]
        y_plot = y[:, 0]
    else:
        s_plot = s.reshape(-1)
        y_plot = y.reshape(-1)

    df = pd.DataFrame(
        {
            "x": np.concatenate([X_reduced, X_reduced]),
            "y": np.concatenate([s_plot, y_plot]),
            "Type": ["Generated Sample"] * len(s_plot) + ["True Data"] * len(y_plot),
        }
    )

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="Type",
        title="Generated Samples vs True Data",
        labels={"x": "Reduced Input Dimension", "y": "Output (dim 0)"},
        opacity=0.4,
    )
    fig.show()


def _reduce_dim(
    X: npt.NDArray[np.float64], non_linear: bool
) -> npt.NDArray[np.float64]:
    if X.shape[1] > 1:
        reducer = UMAP(n_components=1) if non_linear else PCA(n_components=1)
        return reducer.fit_transform(X)
    return X
