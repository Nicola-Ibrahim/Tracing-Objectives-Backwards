"""
Conditional Variational Autoencoder (CVAE) Estimator for Inverse Problems.

This module implements a conditional VAE with Gaussian decoder for modeling posterior
distributions p(x|y) in inverse problems where multiple solutions x can explain the
same observation y.

Architecture:
    - Encoder: q(z|y,cond) - amortized posterior
    - Decoder: p(y|z,cond) - Gaussian likelihood
    - Prior: p(z|cond) - learned conditional prior

Training:
    - Variational lower bound (ELBO) optimization
    - β-VAE framework with KL warmup
    - Free bits for preventing posterior collapse
    - Reconstruction (NLL) + KL divergence loss

Key Features:
    - Conditional generation from learned latent space
    - Temperature-controlled sampling diversity
    - β-annealing for stable training
    - Gaussian observation model with learned variance

Reference:
    Sohn et al. "Learning Structured Output Representation using Deep Conditional
    Generative Models" (2015)
"""

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .....domain.modeling.enums.estimator_type import EstimatorTypeEnum
from .....domain.modeling.interfaces.base_estimator import ProbabilisticEstimator

LOG2PI = float(np.log(2.0 * np.pi))


# -------------------- Conditional prior p(z|cond) -------------------- #
class PriorNet(nn.Module):
    """Gaussian conditional prior p(z | cond): returns (mu_p, logvar_p)."""

    def __init__(
        self,
        cond_dim: int,
        latent_dim: int,
        hidden: int = 64,
        min_logvar: float = -4.0,
        max_logvar: float = 2.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(cond_dim, hidden)
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(cond))
        mu = self.fc_mu(h)
        logvar = torch.clamp(
            self.fc_logvar(h), min=self.min_logvar, max=self.max_logvar
        )
        return mu, logvar


# -------------------- Encoder q(z|y,cond) -------------------- #
class CVAEEncoder(nn.Module):
    """Amortized posterior q(z | y, cond) -> (mu_q, logvar_q)."""

    def __init__(
        self, y_dim: int, cond_dim: int, latent_dim: int = 8, hidden: int = 128
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(y_dim + cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)

    def forward(
        self, y: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(torch.cat([y, cond], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)


# -------------------- Decoder p(y|z,cond) (Gaussian) -------------------- #
class CVAEDecoderGaussian(nn.Module):
    """
    Decoder p(y | z, cond) = N(mu(z,cond), diag(exp(logvar(z,cond)))).
    Returns (mu, logvar) of shape (B, Dy).
    """

    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        y_dim: int,
        hidden: int = 128,
        min_logvar: float = -6.0,
        max_logvar: float = 4.0,
    ):
        super().__init__()
        self.y_dim = int(y_dim)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

        in_dim = latent_dim + cond_dim
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, self.y_dim)
        self.logvar_head = nn.Linear(hidden, self.y_dim)

    @property
    def type(self) -> str:
        return "CVAE-Gaussian"

    def forward(
        self, z: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(torch.cat([z, cond], dim=1))
        mu = self.mu_head(h)  # (B, Dy)
        logvar = self.logvar_head(h)  # (B, Dy)
        logvar = torch.clamp(logvar, min=self.min_logvar, max=self.max_logvar)
        return mu, logvar

    @staticmethod
    def gaussian_nll(
        y: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Mean negative log-likelihood under N(mu, diag(exp(logvar)))."""
        inv_var = torch.exp(-logvar)
        nll_per_sample = 0.5 * (
            torch.sum(logvar + (y - mu) ** 2 * inv_var, dim=-1) + y.size(-1) * LOG2PI
        )
        return torch.mean(nll_per_sample)

    @staticmethod
    def sample(
        mu: torch.Tensor,
        logvar: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """One draw per row from N(mu, diag(exp(logvar)))."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(
            std.shape,
            device=std.device,
            dtype=std.dtype,
            generator=generator,
        )
        return mu + std * eps


# -------------------- CVAE (Gaussian) Estimator -------------------- #
class CVAEEstimator(ProbabilisticEstimator):
    r"""
    Conditional VAE with Gaussian decoder: p(y | z, cond).

    Optimizes:
        E_{q(z|y,cond)}[ -log p(y|z,cond) ] + β * KL[ q(z|y,cond) || p(z|cond) ].

    ───────────────────────────── Workflow ─────────────────────────────
    Training:
        (y, cond) ─▶ q(z|y,cond) → (μ_q, logσ_q²)
                       │
                       ├─ reparam: z = μ_q + σ_q ⊙ ε
                       │
        cond ─▶ p(z|cond) → (μ_p, logσ_p²)
                       │
        (z, cond) ─▶ p(y|z,cond) → (μ_y, logσ_y²)
                       │
            Loss = NLL(y; μ_y, σ_y²) + β * KL[q || p]

    Inference (given cond only):
        cond ─▶ p(z|cond)
            • sample():       z ~ N(μ_p, σ_p²·T) → y ~ N(μ_y(z,cond), σ_y²)
            • infer_mean():   z_s ~ N(μ_p, σ_p²) ; return avg μ_y(z_s,cond)
            • infer_map():    z = μ_p ; return μ_y(μ_p,cond)
            • infer_median(): draw y-samples; return elementwise medians
    ───────────────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        latent_dim: int = 8,
        learning_rate: float = 1e-3,
        beta: float = 0.1,
        kl_warmup: int = 100,
        free_nats: float = 0.0,
        hidden: int = 128,
        decoder_min_logvar: float = -6.0,
        decoder_max_logvar: float = 4.0,
        prior_min_logvar: float = -4.0,
        prior_max_logvar: float = 2.0,
        epochs: int = 200,
        batch_size: int = 128,
        val_size: float = 0.2,
        random_state: int = 42,
    ):
        super().__init__()
        self._latent_dim = latent_dim
        self._learning_rate = learning_rate
        self.beta = beta
        self.beta_final = float(beta)
        self.kl_warmup = kl_warmup
        self.free_nats = free_nats
        self._hidden = hidden
        self._dec_min_lv = decoder_min_logvar
        self._dec_max_lv = decoder_max_logvar
        self._prior_min_lv = prior_min_logvar
        self._prior_max_lv = prior_max_logvar

        self._encoder: CVAEEncoder
        self._decoder: CVAEDecoderGaussian
        self._prior_net: PriorNet

        self._cond_dim: int | None = None
        self._y_dim: int | None = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epochs = epochs
        self.batch_size = batch_size
        self.val_size = val_size
        self.random_state = random_state

    @property
    def type(self) -> str:
        """Short tag for UI/metadata."""
        return getattr(EstimatorTypeEnum, "CVAE", EstimatorTypeEnum.CVAE).value

    # -------------------- data -------------------- #
    def _prepare_dataloaders(
        self,
        cond: NDArray[np.float64],
        y: NDArray[np.float64],
    ):
        """Split into train/val and wrap as DataLoaders."""
        cond_tr, cond_val, y_tr, y_val = train_test_split(
            cond, y, test_size=self.val_size, random_state=self.random_state
        )
        to_t = lambda a: torch.tensor(a, dtype=torch.float32)
        train_loader = DataLoader(
            TensorDataset(to_t(cond_tr), to_t(y_tr)),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(to_t(cond_val), to_t(y_val)),
            batch_size=self.batch_size,
            shuffle=False,
        )
        return train_loader, val_loader

    # -------------------- training -------------------- #
    def fit(
        self,
        cond: NDArray[np.float64] = None,
        y: NDArray[np.float64] = None,
        **kwargs,
    ) -> None:
        """
        Train the CVAE.

        Args
        ----
        cond : array, shape (n_samples, cond_dim)
            Conditioning inputs (e.g. objectives).
        y : array, shape (n_samples, y_dim)
            Targets (e.g. decisions).
        **kwargs :
            For backward compatibility. If `X` is provided, it is treated as `cond`.
        """
        # Back-compat shim
        if cond is None and "X" in kwargs:
            warnings.warn(
                "CVAEEstimator.fit: argument 'X' is deprecated; use 'cond' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            cond = kwargs.pop("X")
        if cond is None or y is None:
            raise TypeError("fit() requires arguments: cond, y")

        # Base validation (sets _X_dim/_y_dim in BaseEstimator)
        super().fit(cond, y)

        self._cond_dim = int(cond.shape[1])
        self._y_dim = int(y.shape[1])

        # Networks
        self._encoder = CVAEEncoder(
            y_dim=self._y_dim,
            cond_dim=self._cond_dim,
            latent_dim=self._latent_dim,
            hidden=self._hidden,
        ).to(self.device)

        self._decoder = CVAEDecoderGaussian(
            latent_dim=self._latent_dim,
            cond_dim=self._cond_dim,
            y_dim=self._y_dim,
            hidden=self._hidden,
            min_logvar=self._dec_min_lv,
            max_logvar=self._dec_max_lv,
        ).to(self.device)

        self._prior_net = PriorNet(
            cond_dim=self._cond_dim,
            latent_dim=self._latent_dim,
            hidden=self._hidden,
            min_logvar=self._prior_min_lv,
            max_logvar=self._prior_max_lv,
        ).to(self.device)

        train_loader, val_loader = self._prepare_dataloaders(cond, y)

        params = (
            list(self._encoder.parameters())
            + list(self._decoder.parameters())
            + list(self._prior_net.parameters())
        )
        opt = torch.optim.Adam(params, lr=self._learning_rate)

        def kl_gaussians(mu_q, logvar_q, mu_p, logvar_p):
            """Closed-form KL for diagonal Gaussians (elementwise → sum)."""
            kl_per_dim = 0.5 * (
                (logvar_p - logvar_q)
                + (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p)
                - 1.0
            )
            if self.free_nats > 0.0:
                kl_per_dim = torch.clamp(
                    kl_per_dim, min=self.free_nats / self._latent_dim
                )
            return torch.mean(torch.sum(kl_per_dim, dim=1))

        for epoch in range(1, self.epochs + 1):
            # β-warmup
            beta = self.beta_final * (
                min(1.0, epoch / max(1, self.kl_warmup)) if self.kl_warmup > 0 else 1.0
            )

            # ---- train ----
            self._encoder.train()
            self._decoder.train()
            self._prior_net.train()
            train_nll_sum = 0.0
            train_kl_sum = 0.0

            for cond_b, y_b in train_loader:
                cond_b = cond_b.to(self.device)
                y_b = y_b.to(self.device)
                opt.zero_grad()

                # q(z|y,cond) via reparameterization
                mu_q, logvar_q = self._encoder(y_b, cond_b)
                std_q = torch.exp(0.5 * logvar_q)
                z = mu_q + std_q * torch.randn_like(std_q)

                # p(z|cond)
                mu_p, logvar_p = self._prior_net(cond_b)

                # p(y|z,cond): Gaussian NLL recon
                mu_y, logvar_y = self._decoder(z, cond_b)
                recon_nll = CVAEDecoderGaussian.gaussian_nll(y_b, mu_y, logvar_y)
                kl = kl_gaussians(mu_q, logvar_q, mu_p, logvar_p)

                base_loss = recon_nll + beta * kl

                loss = base_loss
                loss.backward()
                opt.step()

                train_nll_sum += float(recon_nll.item())
                train_kl_sum += float(kl.item())

            avg_train_nll = train_nll_sum / max(1, len(train_loader))
            avg_train_kl = train_kl_sum / max(1, len(train_loader))
            avg_train = avg_train_nll + beta * avg_train_kl

            # ---- validate ----
            self._encoder.eval()
            self._decoder.eval()
            self._prior_net.eval()
            val_nll_sum = 0.0
            val_kl_sum = 0.0

            with torch.no_grad():
                for cond_v, y_v in val_loader:
                    cond_v = cond_v.to(self.device)
                    y_v = y_v.to(self.device)

                    mu_q_v, logvar_q_v = self._encoder(y_v, cond_v)
                    z_v = mu_q_v + torch.exp(0.5 * logvar_q_v) * torch.randn_like(
                        logvar_q_v
                    )
                    mu_p_v, logvar_p_v = self._prior_net(cond_v)

                    mu_y_v, logvar_y_v = self._decoder(z_v, cond_v)
                    recon_nll_v = CVAEDecoderGaussian.gaussian_nll(
                        y_v, mu_y_v, logvar_y_v
                    )
                    kl_v = kl_gaussians(mu_q_v, logvar_q_v, mu_p_v, logvar_p_v)

                    base_val = recon_nll_v + beta * kl_v

                    total_val = base_val

                    val_nll_sum += float(recon_nll_v.item())
                    val_kl_sum += float(kl_v.item())

            avg_val_nll = val_nll_sum / max(1, len(val_loader))
            avg_val_kl = val_kl_sum / max(1, len(val_loader))
            avg_val = avg_val_nll + beta * avg_val_kl

            # Log everything: total loss + decomposed parts
            self._training_history.log(
                epoch=epoch,
                train_loss=avg_train,
                val_loss=avg_val,
                train_recon=avg_train_nll,
                train_kl=avg_train_kl,
                val_recon=avg_val_nll,
                val_kl=avg_val_kl,
                beta=beta,
            )

    # -------------------- latent helper -------------------- #
    def _sample_z_given_cond(
        self,
        cond_tensor: torch.Tensor,
        n_samples: int,
        temperature: float,
        seed: int | None,
    ) -> torch.Tensor:
        """
        Draw z ~ p(z|cond) with Gaussian tempering: std_T = sqrt(T) * std.
        Returns: (n, n_samples, latent_dim).
        """
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")

        n = cond_tensor.shape[0]
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))

        mu_p, logvar_p = self._prior_net(cond_tensor)  # (n, L)
        std_p = torch.exp(0.5 * logvar_p)  # (n, L)
        t_std = std_p * np.sqrt(float(temperature))  # (n, L)

        eps = torch.randn(
            (n, n_samples, self._latent_dim),
            generator=gen,
            device=self.device,
        )
        return mu_p.unsqueeze(1) + t_std.unsqueeze(1) * eps  # (n, S, L)

    # ---------------- public API (ProbabilisticEstimator) ---------------- #
    def sample(
        self,
        cond: NDArray[np.float64] = None,
        n_samples: int = 200,
        seed: int = 43,
        temperature: float = 1.0,
        max_outputs_per_chunk: int = 20_000,
        **kwargs,
    ) -> NDArray[np.float64]:
        """
        Draw samples from p(y|cond) the conditional way:
            z ~ p(z|cond), then y ~ p(y|z,cond).
        Back-compat: if called with X=..., we map it to cond.
        """
        if cond is None and "X" in kwargs:
            warnings.warn(
                "CVAEEstimator.sample: argument 'X' is deprecated; use 'cond' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            cond = kwargs.pop("X")
        if cond is None:
            raise TypeError("sample() requires argument: cond")

        if self._decoder is None:
            raise RuntimeError("Model not fitted.")
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")

        self._decoder.eval()
        self._prior_net.eval()

        cond_t = torch.tensor(cond, dtype=torch.float32, device=self.device)
        n = cond_t.shape[0]
        S = int(n_samples)
        Dy = int(self._y_dim)

        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        z = self._sample_z_given_cond(
            cond_t, n_samples=S, temperature=temperature, seed=seed
        )  # (n,S,L)

        total = n * S
        z_flat = z.reshape(total, self._latent_dim)
        cond_rep = (
            cond_t.unsqueeze(1)
            .expand(n, S, self._cond_dim)
            .reshape(total, self._cond_dim)
        )

        with torch.no_grad():
            if total <= max_outputs_per_chunk:
                mu, logvar = self._decoder(z_flat, cond_rep)  # (T,Dy)
                y_flat = CVAEDecoderGaussian.sample(mu, logvar)
            else:
                chunks = []
                for start in range(0, total, max_outputs_per_chunk):
                    end = min(total, start + max_outputs_per_chunk)
                    mu_c, lv_c = self._decoder(z_flat[start:end], cond_rep[start:end])
                    y_c = CVAEDecoderGaussian.sample(mu_c, lv_c)
                    chunks.append(y_c)
                y_flat = torch.cat(chunks, dim=0)

        out = y_flat.view(n, S, Dy).cpu().numpy().astype(np.float64, copy=False)
        return out[:, 0, :] if S == 1 else out
