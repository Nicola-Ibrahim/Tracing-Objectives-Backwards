import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .....domain.modeling.enums.estimator_type import EstimatorTypeEnum
from .....domain.modeling.interfaces.base_estimator import (
    ProbabilisticEstimator,
    TrainingHistory,
)

LOG2PI = float(np.log(2.0 * np.pi))


# -------------------- Conditional prior p(z | cond) -------------------- #
class PriorNet(nn.Module):
    """
    Conditional prior p(z | cond): maps condition -> (mu_p, logvar_p).
    Clamps log-variance for numerical stability.
    """

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


# -------------------- Encoder q(z | y, cond) -------------------- #
class CVAEEncoder(nn.Module):
    """Amortized posterior q(z | y, cond) -> (mu_q, logvar_q). Used only in training."""

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


# -------------------- MDN decoder p(y | z, cond) -------------------- #
class CVAEDecoderMDN(nn.Module):
    """
    Mixture Density Network decoder:
      p(y | z, cond) = sum_{k=1..K} pi_k * N(y; mu_k, diag(exp(logvar_k))).
    """

    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        y_dim: int,
        n_components: int = 5,
        hidden: int = 128,
        min_logvar: float = -6.0,
        max_logvar: float = 4.0,
    ):
        super().__init__()
        self.y_dim = int(y_dim)
        self.K = int(n_components)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

        in_dim = latent_dim + cond_dim
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.pi_head = nn.Linear(hidden, self.K)
        self.mu_head = nn.Linear(hidden, self.K * self.y_dim)
        self.logvar_head = nn.Linear(hidden, self.K * self.y_dim)

    @property
    def type(self) -> str:
        return getattr(EstimatorTypeEnum, "CVAE_MDN", EstimatorTypeEnum.CVAE_MDN).value

    def forward(self, z: torch.Tensor, cond: torch.Tensor):
        h = self.backbone(torch.cat([z, cond], dim=1))
        pi_logits = self.pi_head(h)  # (B, K)
        mu = self.mu_head(h).view(-1, self.K, self.y_dim)  # (B, K, Dy)
        logvar = self.logvar_head(h).view(-1, self.K, self.y_dim)
        logvar = torch.clamp(logvar, min=self.min_logvar, max=self.max_logvar)
        return pi_logits, mu, logvar

    @staticmethod
    def mdn_nll(
        y: torch.Tensor, pi_logits: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Negative log-likelihood under a diagonal-Gaussian mixture:
          -log sum_k softmax(pi_logits)_k * N(y; mu_k, diag(exp(logvar_k))).
        Uses log-sum-exp for stability.
        """
        y = y.unsqueeze(1)  # (B, 1, Dy) for broadcast across K
        log_pi = F.log_softmax(pi_logits, dim=-1)  # (B, K)
        inv_var = torch.exp(-logvar)  # (B, K, Dy)
        # Component log-probabilities: (B, K)
        log_prob = -0.5 * (
            torch.sum(logvar + (y - mu) ** 2 * inv_var, dim=-1) + mu.size(-1) * LOG2PI
        )
        # Log mixture
        log_mix = torch.logsumexp(log_pi + log_prob, dim=-1)  # (B,)
        return -torch.mean(log_mix)

    def sample_from_components(
        self, pi_logits: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Draw one y per row:
          k ~ Categorical(softmax(pi_logits)), then y ~ N(mu_k, diag(exp(logvar_k))).
        """
        with torch.no_grad():
            pi = F.softmax(pi_logits, dim=-1)  # (B, K)
            k = torch.multinomial(pi, num_samples=1).squeeze(-1)  # (B,)
            idx = k.view(-1, 1, 1).expand(-1, 1, mu.size(-1))
            sel_mu = torch.gather(mu, 1, idx).squeeze(1)  # (B, Dy)
            sel_lv = torch.gather(logvar, 1, idx).squeeze(1)  # (B, Dy)
            sel_std = torch.exp(0.5 * sel_lv)
            eps = torch.randn_like(sel_mu)
            return sel_mu + sel_std * eps  # (B, Dy)


# -------------------- CVAE Estimator (MDN decoder) -------------------- #
class CVAEMDNEstimator(ProbabilisticEstimator):
    r"""
    Conditional VAE with an MDN decoder for multi-modal p(y | cond).

    Training minimizes (negative ELBO):
        NLL_MDN(y | z, cond) + β * KL[q(z|y,cond) || p(z|cond)].

    Inference (given cond only):
        - sample():      z ~ p(z|cond), then draw y from MDN
        - infer_mean():  MC over z; for each (z,cond), take mixture mean Σ_k softmax(π)_k μ_k and average over draws
        - infer_map():   z := μ_p(cond); take component with max weight and return its μ_k
        - infer_median(): draw y samples and take elementwise medians
    """

    def __init__(
        self,
        latent_dim: int = 8,
        learning_rate: float = 1e-3,
        n_components: int = 5,
        beta: float = 0.1,
        kl_warmup: int = 100,
        free_nats: float = 0.0,
        hidden: int = 128,
        prior_min_logvar: float = -4.0,
        prior_max_logvar: float = 2.0,
        decoder_min_logvar: float = -6.0,
        decoder_max_logvar: float = 4.0,
        epochs: int = 200,
        batch_size: int = 128,
        val_size: float = 0.2,
        random_state: int = 42,
    ):
        super().__init__()
        # core hyperparams
        self._latent_dim = int(latent_dim)
        self._learning_rate = float(learning_rate)
        self._n_components = int(n_components)
        self.beta = float(beta)
        self.beta_final = float(beta)
        self.kl_warmup = int(kl_warmup)
        self.free_nats = float(free_nats)
        self._hidden = int(hidden)
        self._prior_min_lv = float(prior_min_logvar)
        self._prior_max_lv = float(prior_max_logvar)
        self._dec_min_lv = float(decoder_min_logvar)
        self._dec_max_lv = float(decoder_max_logvar)

        # modules
        self._encoder: CVAEEncoder
        self._decoder: CVAEDecoderMDN
        self._prior_net: PriorNet

        # dims
        self._cond_dim: int | None = None
        self._y_dim: int | None = None
        # keep BaseEstimator-compatible alias for dimensionality property
        self._X_dim: int | None = None

        # runtime
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.val_size = float(val_size)
        self.random_state = int(random_state)

    @property
    def type(self) -> str:
        return getattr(EstimatorTypeEnum, "CVAE_MDN", EstimatorTypeEnum.CVAE_MDN).value

    # ---------------- data ---------------- #
    def _prepare_dataloaders(
        self,
        cond: NDArray[np.float64],
        y: NDArray[np.float64],
    ):
        """Train/val split + torch DataLoaders."""
        c_tr, c_val, y_tr, y_val = train_test_split(
            cond, y, test_size=self.val_size, random_state=self.random_state
        )
        to_t = lambda a: torch.tensor(a, dtype=torch.float32)
        train_loader = DataLoader(
            TensorDataset(to_t(c_tr), to_t(y_tr)),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(to_t(c_val), to_t(y_val)),
            batch_size=self.batch_size,
            shuffle=False,
        )
        return train_loader, val_loader

    # ---------------- training ---------------- #
    def fit(self, cond: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Optimize ELBO with MDN reconstruction term and β-KL regularization."""
        super().fit(cond, y)

        self._cond_dim = int(cond.shape[1])
        self._y_dim = int(y.shape[1])
        self._X_dim = self._cond_dim  # for BaseEstimator.dimensionality

        # build modules
        self._encoder = CVAEEncoder(
            y_dim=self._y_dim,
            cond_dim=self._cond_dim,
            latent_dim=self._latent_dim,
            hidden=self._hidden,
        ).to(self.device)
        self._decoder = CVAEDecoderMDN(
            latent_dim=self._latent_dim,
            cond_dim=self._cond_dim,
            y_dim=self._y_dim,
            n_components=self._n_components,
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

        self._training_history = TrainingHistory()

        def kl_gaussians(mu_q, logvar_q, mu_p, logvar_p):
            """Closed-form KL[q||p] for diagonal Gaussians, with optional free-bits."""
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

                # MDN recon term
                pi_logits, mu, logvar = self._decoder(z, cond_b)
                recon_nll = CVAEDecoderMDN.mdn_nll(y_b, pi_logits, mu, logvar)

                # KL regularizer
                kl = kl_gaussians(mu_q, logvar_q, mu_p, logvar_p)

                loss = recon_nll + beta * kl
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

                    pi_logits_v, mu_v, logvar_v = self._decoder(z_v, cond_v)
                    recon_nll_v = CVAEDecoderMDN.mdn_nll(
                        y_v, pi_logits_v, mu_v, logvar_v
                    )
                    kl_v = kl_gaussians(mu_q_v, logvar_q_v, mu_p_v, logvar_p_v)

                    val_nll_sum += float(recon_nll_v.item())
                    val_kl_sum += float(kl_v.item())

            avg_val_nll = val_nll_sum / max(1, len(val_loader))
            avg_val_kl = val_kl_sum / max(1, len(val_loader))
            avg_val = avg_val_nll + beta * avg_val_kl

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

    # ---------------- latent helper ---------------- #
    def _sample_z_given_cond(
        self,
        cond_tensor: torch.Tensor,
        n_samples: int,
        temperature: float,
        seed: int | None,
    ) -> torch.Tensor:
        """
        Draw z ~ p(z|cond) with variance tempering: std_T = sqrt(T) * std  (so Var ∝ T).
        Returns (n, n_samples, latent_dim).
        """
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        n = cond_tensor.shape[0]

        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(seed))

        mu_p, logvar_p = self._prior_net(cond_tensor)  # (n, L)
        std_p = torch.exp(0.5 * logvar_p)  # (n, L)
        t_std = std_p * np.sqrt(float(temperature))  # (n, L)

        eps = torch.randn(
            (n, n_samples, self._latent_dim), generator=gen, device=self.device
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
          z ~ p(z|cond), then y ~ MDN(z,cond).
        Back-compat: if called with X=..., map it to cond (emits DeprecationWarning).
        """
        if cond is None and "X" in kwargs:
            warnings.warn(
                "CVAEMDNEstimator.sample: argument 'X' is deprecated; use 'cond' instead.",
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
                pi_logits, mu, logvar = self._decoder(z_flat, cond_rep)
                y_flat = self._decoder.sample_from_components(
                    pi_logits, mu, logvar
                )  # (T, Dy)
            else:
                outs = []
                for start in range(0, total, max_outputs_per_chunk):
                    end = min(total, start + max_outputs_per_chunk)
                    pi_l, mu_c, lv_c = self._decoder(
                        z_flat[start:end], cond_rep[start:end]
                    )
                    outs.append(self._decoder.sample_from_components(pi_l, mu_c, lv_c))
                y_flat = torch.cat(outs, dim=0)

        out = y_flat.view(n, S, Dy).cpu().numpy().astype(np.float64, copy=False)
        return out[:, 0, :] if S == 1 else out

    def infer_mean(
        self, cond: NDArray[np.float64] = None, **kwargs
    ) -> NDArray[np.float64]:
        """
        Estimate E[y|cond] without observation sampling:
          Draw S=256 latents z ~ p(z|cond); for each, take mixture mean Σ_k softmax(π)_k μ_k; average over z-samples.
        Back-compat: 'X=...' is mapped to 'cond'.
        """
        if cond is None and "X" in kwargs:
            warnings.warn(
                "CVAEMDNEstimator.infer_mean: argument 'X' is deprecated; use 'cond' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            cond = kwargs.pop("X")
        if cond is None:
            raise TypeError("infer_mean() requires argument: cond")

        if self._decoder is None:
            raise RuntimeError("Model not fitted.")

        self._decoder.eval()
        self._prior_net.eval()

        cond_t = torch.tensor(cond, dtype=torch.float32, device=self.device)
        n = cond_t.shape[0]
        Dy = int(self._y_dim)
        S = 256
        temperature = 1.0
        seed = 43
        max_outputs_per_chunk = 20_000

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        with torch.no_grad():
            mu_p, logvar_p = self._prior_net(cond_t)
            std_p = torch.exp(0.5 * logvar_p)
            eps = torch.randn((n, S, self._latent_dim), device=self.device)
            z = mu_p.unsqueeze(1) + float(temperature) * std_p.unsqueeze(1) * eps

            total = n * S
            z_flat = z.reshape(total, self._latent_dim)
            cond_rep = (
                cond_t.unsqueeze(1)
                .expand(n, S, self._cond_dim)
                .reshape(total, self._cond_dim)
            )

            def _mixture_mean_slice(start: int, end: int) -> torch.Tensor:
                pi_logits, mu, _ = self._decoder(z_flat[start:end], cond_rep[start:end])
                pi = torch.softmax(pi_logits, dim=1)  # (chunk, K)
                return torch.sum(pi.unsqueeze(-1) * mu, dim=1)  # (chunk, Dy)

            if total <= max_outputs_per_chunk:
                mix_mean_flat = _mixture_mean_slice(0, total)
            else:
                chunks = []
                for s in range(0, total, max_outputs_per_chunk):
                    e = min(total, s + max_outputs_per_chunk)
                    chunks.append(_mixture_mean_slice(s, e))
                mix_mean_flat = torch.cat(chunks, dim=0)

            mix_mean = mix_mean_flat.view(n, S, Dy)
            out = mix_mean.mean(dim=1)  # (n, Dy)

        return out.cpu().numpy().astype(np.float64, copy=False)

    def infer_median(
        self, cond: NDArray[np.float64] = None, **kwargs
    ) -> NDArray[np.float64]:
        """
        Approximate posterior median: draw S=501 y-samples from p(y|cond) and take elementwise medians.
        Back-compat: 'X=...' is mapped to 'cond'.
        """
        if cond is None and "X" in kwargs:
            warnings.warn(
                "CVAEMDNEstimator.infer_median: argument 'X' is deprecated; use 'cond' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            cond = kwargs.pop("X")
        if cond is None:
            raise TypeError("infer_median() requires argument: cond")

        draws = self.sample(
            cond=cond,
            n_samples=501,
            seed=43,
            temperature=1.0,
            max_outputs_per_chunk=20_000,
        )
        if draws.ndim == 2:
            return draws.astype(np.float64, copy=False)
        med = np.median(draws, axis=1)
        return med.astype(np.float64, copy=False)

    def infer_map(
        self, cond: NDArray[np.float64] = None, **kwargs
    ) -> NDArray[np.float64]:
        """
        Deterministic, MAP-like point at z := μ_p(cond):
          choose component k* with highest weight and return μ_{k*}.
        Back-compat: 'X=...' is mapped to 'cond'.
        """
        if cond is None and "X" in kwargs:
            warnings.warn(
                "CVAEMDNEstimator.infer_map: argument 'X' is deprecated; use 'cond' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            cond = kwargs.pop("X")
        if cond is None:
            raise TypeError("infer_map() requires argument: cond")

        if self._decoder is None:
            raise RuntimeError("Model not fitted.")

        self._decoder.eval()
        self._prior_net.eval()

        cond_t = torch.tensor(cond, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mu_p, _ = self._prior_net(cond_t)  # (n, L)
            pi_logits, mu, _ = self._decoder(mu_p, cond_t)  # (n, K), (n, K, Dy)
            pi = torch.softmax(pi_logits, dim=1)  # (n, K)
            k_star = torch.argmax(pi, dim=1)  # (n,)
            out = mu[torch.arange(mu.size(0), device=mu.device), k_star, :]  # (n, Dy)

        return out.cpu().numpy().astype(np.float64, copy=False)
