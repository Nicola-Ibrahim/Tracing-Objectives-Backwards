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


# -------------------- Conditional prior network -------------------- #
class PriorNet(nn.Module):
    """
    Conditional prior p(z | X) with clamped log-variance for stability.

    Maps condition X (inputs) -> (μ_p(X), log σ_p^2(X)).
    Used in training (KL term) and inference (sampling z | X).
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

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(X))
        mu = self.fc_mu(h)
        # Clamp variance range for numerical stability.
        logvar = torch.clamp(
            self.fc_logvar(h), min=self.min_logvar, max=self.max_logvar
        )
        return mu, logvar


# -------------------- Encoder -------------------- #
class CVAEEncoder(nn.Module):
    """
    Amortized posterior q(z | y, X) -> (μ_q, log σ_q^2).

    Used ONLY during training: encodes target y with condition X to infer z.
    """

    def __init__(self, y_dim: int, X_dim: int, latent_dim: int = 8, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(y_dim + X_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)

    def forward(
        self, y: torch.Tensor, X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Concatenate y (target) and X (condition) to infer z.
        h = self.net(torch.cat([y, X], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)


# -------------------- MDN decoder -------------------- #
class CVAEDecoderMDN(nn.Module):
    """
    Decoder p(y | z, X) as a K-component diagonal-Gaussian mixture.

    Given (z, X), predicts mixture logits π, component means μ, and log-variances.
    """

    def __init__(
        self,
        latent_dim: int,
        X_dim: int,
        y_dim: int,
        n_components: int = 5,
        hidden: int = 128,
        min_logvar: float = -6.0,
        max_logvar: float = 4.0,
    ):
        super().__init__()
        self.y_dim = int(y_dim)  # output dimensionality
        self.K = int(n_components)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

        in_dim = latent_dim + X_dim
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # mixture parameters
        self.pi_head = nn.Linear(hidden, self.K)
        self.mu_head = nn.Linear(hidden, self.K * self.y_dim)
        self.logvar_head = nn.Linear(hidden, self.K * self.y_dim)

    @property
    def type(self) -> str:
        return getattr(EstimatorTypeEnum, "CVAE_MDN", EstimatorTypeEnum.CVAE_MDN).value

    def forward(self, z: torch.Tensor, X: torch.Tensor):
        # Predict mixture parameters π, μ, σ^2 for each (z, X).
        h = self.backbone(torch.cat([z, X], dim=1))
        pi_logits = self.pi_head(h)  # (B, K)
        mu = self.mu_head(h).view(-1, self.K, self.y_dim)  # (B, K, Dy)
        logvar = self.logvar_head(h).view(-1, self.K, self.y_dim)
        # Clamp for stable likelihood and gradients.
        logvar = torch.clamp(logvar, min=self.min_logvar, max=self.max_logvar)
        return pi_logits, mu, logvar

    @staticmethod
    def mdn_nll(
        y: torch.Tensor, pi_logits: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Negative log-likelihood of y under a diagonal-Gaussian mixture.

        Computes -log ∑_k π_k N(y; μ_k, σ_k^2) using log-sum-exp for stability.
        """
        # y: (B, Dy) -> (B, 1, Dy) to broadcast across K.
        y = y.unsqueeze(1)
        log_pi = F.log_softmax(pi_logits, dim=-1)  # (B, K)
        inv_var = torch.exp(-logvar)
        # Component log-prob: log N(y; μ, σ^2) → (B, K)
        log_prob = -0.5 * (
            torch.sum(logvar + (y - mu) ** 2 * inv_var, dim=-1) + mu.shape[-1] * LOG2PI
        )
        # log ∑_k exp(log π_k + log N_k)
        log_mix = torch.logsumexp(log_pi + log_prob, dim=-1)  # (B,)
        return -torch.mean(log_mix)

    def sample_from_components(
        self, pi_logits: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Draw one sample per row from the mixture.

        Steps: sample component index k ~ Cat(softmax(π)); then y ~ N(μ_k, σ_k^2).
        """
        with torch.no_grad():
            pi = F.softmax(pi_logits, dim=-1)  # (B, K)
            comp = torch.multinomial(pi, num_samples=1).squeeze(-1)  # (B,)
            idx = comp.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, mu.size(-1))
            sel_mu = torch.gather(mu, 1, idx).squeeze(1)  # (B, Dy)
            sel_std = torch.exp(0.5 * torch.gather(logvar, 1, idx).squeeze(1))
            eps = torch.randn_like(sel_mu)
            return sel_mu + sel_std * eps


# -------------------- CVAE Estimator (MDN decoder) -------------------- #
class CVAEMDNEstimator(ProbabilisticEstimator):
    """
    CVAE with MDN decoder for multi-modal p(y | X).

    Training minimizes:  NLL(y | z, X)  +  β * KL[q(z|y,X) || p(z|X)]
      - Reconstruction: MDN negative log-likelihood (not MSE).
      - KL: aligns encoder posterior with conditional prior; warm-up & free-bits supported.
    """

    def __init__(
        self,
        latent_dim: int = 8,
        learning_rate: float = 1e-3,
        n_components: int = 5,
        beta: float = 0.1,
        kl_warmup: int = 100,
        free_nats: float = 0.0,
        epochs: int = 200,
        batch_size: int = 128,
        val_size: float = 0.2,
        random_state: int = 42,
    ):
        super().__init__()
        self._latent_dim = int(latent_dim)
        self._learning_rate = float(learning_rate)
        self._n_components = int(n_components)
        self.beta = float(beta)
        self.beta_final = float(beta)
        self.kl_warmup = int(kl_warmup)
        self.free_nats = float(free_nats)

        self._encoder: CVAEEncoder
        self._decoder: CVAEDecoderMDN
        self._prior_net: PriorNet

        self._X_dim: int | None = None  # input/condition dimension
        self._y_dim: int | None = None  # output/target dimension
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epochs = epochs
        self.batch_size = batch_size
        self.val_size = val_size
        self.random_state = random_state

    @property
    def type(self) -> str:
        return getattr(EstimatorTypeEnum, "CVAE_MDN", EstimatorTypeEnum.CVAE_MDN).value

    # ---- data ---- #
    def _prepare_dataloaders(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ):
        # Standard train/val split on numpy arrays, then wrap to torch.
        Xtr, Xval, ytr, yval = train_test_split(
            X, y, test_size=self.val_size, random_state=self.random_state
        )
        to_t = lambda a: torch.tensor(a, dtype=torch.float32)
        train_loader = DataLoader(
            TensorDataset(to_t(Xtr), to_t(ytr)),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(to_t(Xval), to_t(yval)),
            batch_size=self.batch_size,
            shuffle=False,
        )
        return train_loader, val_loader

    # ---- training ---- #
    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Fit the CVAE-MDN using the configured training hyperparameters."""
        super().fit(X, y)

        # Shapes: X is condition/input, y is target/output.
        self._X_dim = int(X.shape[1])
        self._y_dim = int(y.shape[1])

        # ---- model components ----
        self._encoder = CVAEEncoder(
            y_dim=self._y_dim, X_dim=self._X_dim, latent_dim=self._latent_dim
        ).to(self.device)

        self._decoder = CVAEDecoderMDN(
            latent_dim=self._latent_dim,
            X_dim=self._X_dim,
            y_dim=self._y_dim,
            n_components=self._n_components,
        ).to(self.device)

        self._prior_net = PriorNet(
            cond_dim=self._X_dim, latent_dim=self._latent_dim, hidden=128
        ).to(self.device)

        # Data loaders
        train_loader, val_loader = self._prepare_dataloaders(X, y)

        # Single optimizer over encoder, decoder, and prior net.
        params = (
            list(self._encoder.parameters())
            + list(self._decoder.parameters())
            + list(self._prior_net.parameters())
        )
        opt = torch.optim.Adam(params, lr=self._learning_rate)

        # Track loss curves etc.
        self._training_history = TrainingHistory()

        # KL(q||p) for diagonal Gaussians (per-sample mean).
        def kl_gaussians(mu_q, logvar_q, mu_p, logvar_p):
            # Closed-form KL for diag Gaussians: sum over dims, average over batch.
            kl_per_dim = 0.5 * (
                (logvar_p - logvar_q)
                + (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p)
                - 1.0
            )
            # Free-bits (free_nats) prevents KL from collapsing to zero immediately.
            if self.free_nats > 0.0:
                kl_per_dim = torch.clamp(
                    kl_per_dim, min=self.free_nats / self._latent_dim
                )
            return torch.mean(torch.sum(kl_per_dim, dim=1))

        for epoch in range(1, self.epochs + 1):
            # KL warmup: ramp β from 0 → beta_final across `kl_warmup` epochs.
            beta = self.beta_final * (
                min(1.0, epoch / max(1, self.kl_warmup)) if self.kl_warmup > 0 else 1.0
            )

            # ---- train ----
            self._encoder.train()
            self._decoder.train()
            self._prior_net.train()
            train_nll_sum = 0.0
            train_kl_sum = 0.0

            for Xb, yb in train_loader:
                Xb = Xb.to(self.device)  # Xb: conditions/inputs
                yb = yb.to(self.device)  # yb: targets/outputs
                opt.zero_grad()

                # (1) Amortized posterior q(z|y,X)
                mu_q, logvar_q = self._encoder(yb, Xb)
                std_q = torch.exp(0.5 * logvar_q)
                z = mu_q + std_q * torch.randn_like(std_q)  # reparam trick

                # (2) Conditional prior p(z|X)
                mu_p, logvar_p = self._prior_net(Xb)

                # (3) Decoder likelihood p(y|z,X) via MDN
                pi_logits, mu, logvar = self._decoder(z, Xb)

                # Reconstruction term: NLL under mixture (averaged over batch)
                recon_nll = CVAEDecoderMDN.mdn_nll(yb, pi_logits, mu, logvar)
                # Regularization: KL[q(z|y,X) || p(z|X)]
                kl = kl_gaussians(mu_q, logvar_q, mu_p, logvar_p)

                # (4) Total loss = recon + β * KL  (negative ELBO)
                loss = recon_nll + beta * kl
                loss.backward()
                opt.step()

                train_nll_sum += float(recon_nll.item())
                train_kl_sum += float(kl.item())

            # Epoch-level averages (for logging)
            avg_train_nll = train_nll_sum / max(1, len(train_loader))
            avg_train_kl = train_kl_sum / max(1, len(train_loader))
            avg_train = avg_train_nll + beta * avg_train_kl

            # ---- validate (same computations, no gradient) ----
            self._encoder.eval()
            self._decoder.eval()
            self._prior_net.eval()
            val_nll_sum = 0.0
            val_kl_sum = 0.0
            with torch.no_grad():
                for Xv, yv in val_loader:
                    Xv = Xv.to(self.device)
                    yv = yv.to(self.device)
                    mu_q_v, logvar_q_v = self._encoder(yv, Xv)
                    z_v = mu_q_v + torch.exp(0.5 * logvar_q_v) * torch.randn_like(
                        logvar_q_v
                    )
                    mu_p_v, logvar_p_v = self._prior_net(Xv)
                    pi_logits_v, mu_v, logvar_v = self._decoder(z_v, Xv)
                    recon_nll_v = CVAEDecoderMDN.mdn_nll(
                        yv, pi_logits_v, mu_v, logvar_v
                    )
                    kl_v = kl_gaussians(mu_q_v, logvar_q_v, mu_p_v, logvar_p_v)
                    val_nll_sum += float(recon_nll_v.item())
                    val_kl_sum += float(kl_v.item())

            avg_val_nll = val_nll_sum / max(1, len(val_loader))
            avg_val_kl = val_kl_sum / max(1, len(val_loader))
            avg_val = avg_val_nll + beta * avg_val_kl

            # Persist per-epoch metrics (useful for visualizer)
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

    # ---- latent sampling ---- #
    def _sample_z_given_y(
        self,
        X_tensor: torch.Tensor,
        n_samples: int,
        temperature: float,
        seed: int,
    ) -> torch.Tensor:
        """
        Draw S latent samples z ~ p(z|X) for each input X (inference helper).

        Reparameterization using PriorNet stats:
          z = μ_p(X) + (temperature * σ_p(X)) * ε, ε ~ N(0, I)
        """
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")

        n = X_tensor.shape[0]
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))

        mu_p, logvar_p = self._prior_net(X_tensor)  # (n, latent)
        std_p = torch.exp(0.5 * logvar_p)
        eps = torch.randn(
            (n, n_samples, self._latent_dim), generator=gen, device=self.device
        )
        return (
            mu_p.unsqueeze(1) + float(temperature) * std_p.unsqueeze(1) * eps
        )  # (n, S, latent)

    # ---- public API: sample / predict ---- #
    def sample(
        self,
        X: NDArray[np.float64],
        n_samples: int = 200,
        seed: int = 44,
        temperature: float = 1.0,
        use_prior: bool = True,
        max_outputs_per_chunk: int = 20_000,
    ) -> NDArray[np.float64]:
        """
        Draw `n_samples` IID samples per input from the CVAE-implied p(y|X):
        z ~ p(z|X) (or N(0,I) if use_prior=False), then y ~ decoder(z, X) mixture.

        Returns either (n, Dy) if n_samples==1 or (n, n_samples, Dy) otherwise.
        """
        if self._decoder is None:
            raise RuntimeError("Model not fitted.")
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")

        self._decoder.eval()
        self._prior_net.eval()

        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        n = X_tensor.shape[0]
        S = int(n_samples)
        Dy = int(self._y_dim)

        # Sample latent z per input, conditionally (preferred) or unconditionally.
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        if use_prior:
            z = self._sample_z_given_y(
                X_tensor, n_samples=S, temperature=temperature, seed=seed
            )  # (n,S,latent)
        else:
            z = torch.randn((n, S, self._latent_dim), device=self.device) * float(
                temperature
            )

        # Flatten samples and repeat conditions to run decoder in a single pass.
        total = n * S
        z_flat = z.reshape(total, self._latent_dim)  # (T, latent)
        X_rep = (
            X_tensor.unsqueeze(1).expand(n, S, self._X_dim).reshape(total, self._X_dim)
        )  # (T, DX)

        with torch.no_grad():
            # Chunked decoding to bound memory.
            if total <= max_outputs_per_chunk:
                pi_logits, mu, logvar = self._decoder(
                    z_flat, X_rep
                )  # (T,K), (T,K,Dy), (T,K,Dy)
                y_flat = self._decoder.sample_from_components(
                    pi_logits, mu, logvar
                )  # (T, Dy)
            else:
                out_chunks = []
                for start in range(0, total, max_outputs_per_chunk):
                    end = min(total, start + max_outputs_per_chunk)
                    pi_l, mu_c, lv_c = self._decoder(
                        z_flat[start:end], X_rep[start:end]
                    )
                    y_c = self._decoder.sample_from_components(
                        pi_l, mu_c, lv_c
                    )  # (chunk, Dy)
                    out_chunks.append(y_c)
                y_flat = torch.cat(out_chunks, dim=0)

        out = y_flat.view(n, S, Dy).cpu().numpy().astype(np.float64, copy=False)
        return out[:, 0, :] if S == 1 else out

    def predict(
        self,
        X: np.typing.NDArray[np.float64],
        n_samples: int = 1,
        temperature: float = 1.0,
        seed: int = 43,
        use_prior: bool = True,
        *,
        max_outputs_per_chunk: int = 20_000,
    ) -> np.typing.NDArray[np.float64]:
        """
        One **stochastic** draw per input (wrapper over sample with n_samples=1).
        """
        s = self.sample(
            X,
            n_samples=n_samples,
            seed=seed,
            temperature=temperature,
            use_prior=use_prior,
            max_outputs_per_chunk=max_outputs_per_chunk,
        )
        return s.astype(np.float64, copy=False)

    def predict_mean(
        self,
        X: NDArray[np.float64],
        n_samples: int = 256,
        seed: int = 43,
        temperature: float = 1.0,
        deterministic: bool = False,
        max_outputs_per_chunk: int = 20_000,
    ) -> NDArray[np.float64]:
        """
        Conditional mean E[y|X].

        deterministic=True:
          Use a single z = μ_p(X), then return analytic mixture mean Σ_k softmax(π)_k μ_k.
        deterministic=False:
          Monte-Carlo over z ~ p(z|X): take mixture mean per z, then average across S.
        """
        if self._decoder is None:
            raise RuntimeError("Model not fitted.")
        if not deterministic and n_samples < 1:
            raise ValueError("n_samples must be >= 1")

        self._decoder.eval()
        self._prior_net.eval()

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        n = X_t.shape[0]
        Dy = int(self._y_dim)

        # RNG for latent sampling
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        with torch.no_grad():
            # PriorNet provides p(z|X) parameters.
            mu_p, logvar_p = self._prior_net(X_t)  # (n, latent)
            std_p = torch.exp(0.5 * logvar_p)

            if deterministic:
                S = 1
                z = mu_p.unsqueeze(1)  # (n,1,latent)
            else:
                S = int(max(1, n_samples))
                eps = torch.randn((n, S, self._latent_dim), device=self.device)
                z = (
                    mu_p.unsqueeze(1) + float(temperature) * std_p.unsqueeze(1) * eps
                )  # (n,S,latent)

            total = n * S
            z_flat = z.reshape(total, self._latent_dim)
            X_rep = (
                X_t.unsqueeze(1).expand(n, S, self._X_dim).reshape(total, self._X_dim)
            )

            # Decode (z,X) and compute analytic mixture means per row (no component sampling).
            def _mix_means_for_slice(start: int, end: int) -> torch.Tensor:
                pi_logits, mu, logvar = self._decoder(
                    z_flat[start:end], X_rep[start:end]
                )
                pi = torch.softmax(pi_logits, dim=1)  # (chunk, K)
                return torch.sum(pi.unsqueeze(-1) * mu, dim=1)  # (chunk, Dy)

            if total <= max_outputs_per_chunk:
                mix_mean_flat = _mix_means_for_slice(0, total)  # (T, Dy)
            else:
                chunks = []
                for start in range(0, total, max_outputs_per_chunk):
                    end = min(total, start + max_outputs_per_chunk)
                    chunks.append(_mix_means_for_slice(start, end))
                mix_mean_flat = torch.cat(chunks, dim=0)  # (T, Dy)

            mix_mean = mix_mean_flat.view(n, S, Dy)  # (n,S,Dy)
            out = mix_mean.mean(dim=1) if S > 1 else mix_mean.squeeze(1)  # (n,Dy)

        return out.cpu().numpy().astype(np.float64, copy=False)

    def predict_map(
        self,
        X: NDArray[np.float64],
        *,
        seed: int | None = None,
        use_prior: bool = True,
    ) -> NDArray[np.float64]:
        """MAP estimate via the highest-weight mixture component with latent z=μ_p(X)."""

        if self._decoder is None:
            raise RuntimeError("Model not fitted.")

        self._decoder.eval()
        self._prior_net.eval()

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        with torch.no_grad():
            if use_prior:
                mu_p, _ = self._prior_net(X_t)
            else:
                mu_p = torch.zeros((X_t.size(0), self._latent_dim), device=self.device)
            z = mu_p
            pi_logits, mu, _ = self._decoder(z, X_t)
            pi = torch.softmax(pi_logits, dim=1)
            k_star = torch.argmax(pi, dim=1)
            out = mu[torch.arange(mu.size(0), device=mu.device), k_star, :]

        return out.cpu().numpy().astype(np.float64, copy=False)

    def predict_median(
        self,
        X: NDArray[np.float64],
        n_samples: int = 501,
        seed: int = 43,
        *,
        temperature: float = 1.0,
        use_prior: bool = True,
        max_outputs_per_chunk: int = 20_000,
    ) -> NDArray[np.float64]:
        """Approximate posterior median via Monte-Carlo draws from p(y|X)."""

        if self._decoder is None:
            raise RuntimeError("Model not fitted.")
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")

        draws = self.sample(
            X,
            n_samples=n_samples,
            seed=seed,
            temperature=temperature,
            use_prior=use_prior,
            max_outputs_per_chunk=max_outputs_per_chunk,
        )

        if draws.ndim == 2:
            return draws.astype(np.float64, copy=False)

        median = np.median(draws, axis=1)
        return median.astype(np.float64, copy=False)
