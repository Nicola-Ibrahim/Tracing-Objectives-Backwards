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

from ....domain.modeling.enums.estimator_type import EstimatorTypeEnum
from ....domain.modeling.interfaces.base_estimator import (
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
        if distribution_family == DistributionFamilyEnum.NORMAL:
            comp = Normal(mu, sigma)
        elif distribution_family == DistributionFamilyEnum.LAPLACE:
            comp = Laplace(mu, sigma)
        elif distribution_family == DistributionFamilyEnum.LOGNORMAL:
            comp = LogNormal(mu, sigma)
        else:
            raise ValueError(
                f"Unsupported distribution family: {distribution_family.value}"
            )

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
            F.softplus(sigma) + 1e-6
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
      - predict_mean / predict_map / predict_std (analytic where possible)
      - predict_median / predict_quantiles (sampling-based)
      - get_mixture_parameters(X) -> (pi, mu, sigma) as numpy arrays
    """

    def __init__(
        self,
        num_mixtures: int = -1,
        learning_rate: float = 1e-4,
        early_stopping_patience: int = 10,
        distribution_family: DistributionFamilyEnum = DistributionFamilyEnum.NORMAL,
        gmm_boost: bool = False,
        hidden_layers: list[int] = [64],
        hidden_activation_fn_name: ActivationFunctionEnum = ActivationFunctionEnum.RELU,
        optimizer_fn_name: OptimizerFunctionEnum = OptimizerFunctionEnum.ADAM,
        verbose: bool = False,
    ):
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
        self._clusterer: GaussianMixture = None
        self._best_model_state_dict: dict = None

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        batch_size: int = 32,
        test_size: float = 0.2,
    ) -> tuple[DataLoader, DataLoader]:
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
        **kwargs,
    ) -> None:
        """
        Fit the MDN with early stopping.
        kwargs:
            epochs: int = 100
            batch_size: int = 32
            weight_decay: float = 0.0
            clip_grad_norm: Optional[float] = None
            seed: int = None  (for reproducibility)
        """
        super().fit(X, y)

        # Optional reproducibility
        seed = kwargs.get("seed", None)
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

        epochs = int(kwargs.get("epochs", 100))
        batch_size = int(kwargs.get("batch_size", 32))
        weight_decay = float(kwargs.get("weight_decay", 0.0))
        clip_grad_norm = kwargs.get("clip_grad_norm", None)

        self._maybe_fail_on_lognormal(y)
        self._prepare_model(X, y)
        train_loader, val_loader = self._prepare_dataloaders(X, y, batch_size)

        optimizer_fn = self._get_optimizer_fn(self._optimizer_fn_name)
        optimizer = optimizer_fn(
            self._model.parameters(), lr=self._learning_rate, weight_decay=weight_decay
        )

        best_loss = float("inf")
        patience_counter = 0

        # Training loop
        for epoch in range(epochs):
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
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(), float(clip_grad_norm)
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

            # Early stopping
            if avg_val < best_loss:
                best_loss = avg_val
                patience_counter = 0
                self._best_model_state_dict = self._model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self._early_stopping_patience:
                    if self._verbose:
                        print(
                            f"Early stopping at epoch {epoch}. Best val loss: {best_loss:.4f}"
                        )
                    if self._best_model_state_dict:
                        self._model.load_state_dict(self._best_model_state_dict)
                    break

            self._training_history.epochs.append(epoch)
            self._training_history.train_loss.append(float(avg_train))
            self._training_history.val_loss.append(float(avg_val))

        self._model.eval()

    # ----------------------- Inference API -----------------------

    def predict(
        self,
        X: npt.NDArray[np.float64],
        seed: int | None = None,
        *,
        temperature: float = 1.0,
        max_outputs_per_chunk: int = 20_000,
    ) -> npt.NDArray[np.float64]:
        """
        One **stochastic** draw per input from p(x|y).

        Determinism:
        - Stochastic. Use `seed` for reproducibility.
        - `temperature` rescales component scales (σ), not means.

        Shapes:
        - X: (n, in_dim)
        - return: (n, out_dim)
        """
        s = self.sample(
            X,
            n_samples=1,
            seed=seed,
            temperature=temperature,
            max_outputs_per_chunk=max_outputs_per_chunk,
        )
        return s.astype(np.float64, copy=False)

    def sample(
        self,
        X: npt.NDArray[np.float64],
        n_samples: int = 1,
        seed: int | None = None,
        *,
        temperature: float = 1.0,
        max_outputs_per_chunk: int = 20_000,
    ) -> npt.NDArray[np.float64]:
        """
        Draw `n_samples` IID samples per input from p(x|y).

        Shapes:
        - X: (n, in_dim)
        - return:
            (n, out_dim)              if n_samples == 1
            (n, n_samples, out_dim)   if n_samples  > 1
        """
        self._ensure_fitted()
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")

        X_tensor = self._prepare_inputs(X)
        device = self._model_device()
        n = X_tensor.shape[0]
        n_samples = int(n_samples)

        # RNG
        gen = None
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed))

        total_outputs = n * max(1, n_samples)

        # Fast path
        if total_outputs <= max_outputs_per_chunk:
            with torch.no_grad():
                pi, mu, sigma = self._model(X_tensor)
                if temperature != 1.0:
                    sigma = self._apply_temperature(sigma, temperature)
                dist = self._model._get_distribution(
                    pi, mu, sigma, self._distribution_family
                )
                if n_samples == 1:
                    y = dist.sample(generator=gen)  # (n, out)
                    return y.cpu().numpy().astype(np.float64, copy=False)
                else:
                    y = dist.sample((n_samples,), generator=gen)  # (n_s, n, out)
                    return (
                        y.permute(1, 0, 2).cpu().numpy().astype(np.float64, copy=False)
                    )

        # Chunked path
        results = []
        chunk_batch = max(1, max_outputs_per_chunk // max(1, n_samples))
        for start in range(0, n, chunk_batch):
            end = min(n, start + chunk_batch)
            X_chunk = X_tensor[start:end]
            with torch.no_grad():
                pi_c, mu_c, sigma_c = self._model(X_chunk)
                if temperature != 1.0:
                    sigma_c = self._apply_temperature(sigma_c, temperature)
                dist_c = self._model._get_distribution(
                    pi_c, mu_c, sigma_c, self._distribution_family
                )
                if n_samples == 1:
                    s_chunk = dist_c.sample(generator=gen)  # (chunk, out)
                    results.append(s_chunk.cpu())
                else:
                    s_chunk = dist_c.sample(
                        (n_samples,), generator=gen
                    )  # (n_s, chunk, out)
                    s_chunk = s_chunk.permute(1, 0, 2).cpu()  # (chunk, n_s, out)
                    results.append(s_chunk)

        out = torch.cat(results, dim=0).numpy().astype(np.float64, copy=False)
        return out

    def predict_mean(
        self,
        X: npt.NDArray[np.float64],
        n_samples: int = 256,
        seed: int | None = None,
        *,
        temperature: float = 1.0,
        max_outputs_per_chunk: int = 20_000,
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
            n_samples=n_samples,
            seed=seed,
            temperature=temperature,
            max_outputs_per_chunk=max_outputs_per_chunk,
        )
        if s.ndim == 2:
            return s.astype(np.float64, copy=False)  # n_samples==1, trivial mean
        return s.mean(axis=1).astype(np.float64, copy=False)

    def predict_median(
        self,
        X: npt.NDArray[np.float64],
        n_samples: int = 501,
        seed: int | None = None,
        *,
        temperature: float = 1.0,
        max_outputs_per_chunk: int = 20_000,
    ) -> npt.NDArray[np.float64]:
        """
        **Monte-Carlo median** per input (marginal, per output dimension).

        Shapes:
        - X: (n, in_dim)
        - return: (n, out_dim)
        """
        s = self.sample(
            X,
            n_samples=n_samples,
            seed=seed,
            temperature=temperature,
            max_outputs_per_chunk=max_outputs_per_chunk,
        )
        if s.ndim == 2:
            return s.astype(np.float64, copy=False)
        return np.median(s, axis=1).astype(np.float64, copy=False)

    def predict_quantiles(
        self,
        X: npt.NDArray[np.float64],
        qs: Sequence[float] = (0.05, 0.95),
        n_samples: int = 500,
        seed: int | None = None,
        *,
        temperature: float = 1.0,
        max_outputs_per_chunk: int = 20_000,
    ) -> npt.NDArray[np.float64]:
        """
        **Monte-Carlo quantiles** per input (marginal, per output dimension).

        Shapes:
        - X: (n, in_dim)
        - return: (n, len(qs), out_dim)
        """
        s = self.sample(
            X,
            n_samples=n_samples,
            seed=seed,
            temperature=temperature,
            max_outputs_per_chunk=max_outputs_per_chunk,
        )
        if s.ndim == 2:
            s = s[:, None, :]
        percents = [float(q) * 100.0 for q in qs]
        q_arr = np.percentile(s, percents, axis=1)  # (len(qs), n, out)
        return np.transpose(q_arr, (1, 0, 2)).astype(np.float64, copy=False)

    def predict_std(
        self,
        X: npt.NDArray[np.float64],
        n_samples: int = 500,
        seed: int | None = None,
        *,
        temperature: float = 1.0,
        max_outputs_per_chunk: int = 20_000,
    ) -> npt.NDArray[np.float64]:
        """
        **Monte-Carlo standard deviation** per input (marginal, per output dimension).

        Notes:
        - Uses sample variance across draws from p(x|y).
        - Prefer MC here to keep semantics aligned with CVAE and other models.

        Shapes:
        - X: (n, in_dim)
        - return: (n, out_dim)
        """
        s = self.sample(
            X,
            n_samples=n_samples,
            seed=seed,
            temperature=temperature,
            max_outputs_per_chunk=max_outputs_per_chunk,
        )
        if s.ndim == 2:
            # with one sample, std is undefined; return zeros
            return np.zeros_like(s, dtype=np.float64)
        return s.std(axis=1, ddof=0).astype(np.float64, copy=False)

    def predict_map(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        **Deterministic** mean of the **MAP component** per input:
        k* = argmax_k π_k(y); returns μ_{k*}(y).

        Shapes:
        - X: (n, in_dim)
        - return: (n, out_dim)
        """
        self._ensure_fitted()
        X_t = self._prepare_inputs(X)
        with torch.no_grad():
            pi, mu, sigma = self._model(X_t)
            k_star = torch.argmax(pi, dim=1)  # (n,)
            out = mu[torch.arange(mu.size(0), device=mu.device), k_star, :]  # (n, out)
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
        if getattr(self, "_model", None) is None:
            raise RuntimeError("Model not fitted. Call 'fit' first.")

    def _model_device(self) -> torch.device:
        return next(self._model.parameters()).device

    def _prepare_inputs(self, X: npt.NDArray[np.float64]) -> torch.Tensor:
        """
        Convert to torch, optionally append GMM-boost features, and move to device.
        """
        device = self._model_device()
        X_t = torch.tensor(X, dtype=torch.float32, device=device)

        if (
            getattr(self, "_gmm_boost", False)
            and getattr(self, "_clusterer", None) is not None
        ):
            cluster_probs = self._clusterer.predict_proba(X)  # numpy
            cp_t = torch.tensor(cluster_probs, dtype=torch.float32, device=device)
            X_t = torch.cat([X_t, cp_t], dim=1)

        return X_t

    def _apply_temperature(
        self, sigma: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        """
        Temperature scaling for component scales (σ is std/scale).
        temperature > 1.0 increases variance; < 1.0 decreases.
        """
        if temperature == 1.0:
            return sigma
        return sigma * float(temperature)


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
