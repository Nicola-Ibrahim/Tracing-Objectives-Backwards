import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from ....domain.modeling.enums.estimator_type import EstimatorTypeEnum
from ....domain.modeling.interfaces.base_estimator import (
    ProbabilisticEstimator,
    TrainingHistory,
)

LOG2PI = float(np.log(2.0 * np.pi))


# -------------------- Conditional prior p(z|X) -------------------- #
class PriorNet(nn.Module):
    """Gaussian prior p(z | X): returns (mu_p, logvar_p)."""

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
        logvar = torch.clamp(
            self.fc_logvar(h), min=self.min_logvar, max=self.max_logvar
        )
        return mu, logvar


# -------------------- Encoder q(z|y,X) -------------------- #
class CVAEEncoder(nn.Module):
    """Amortized posterior q(z | y, X) -> (mu_q, logvar_q)."""

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
        h = self.net(torch.cat([y, X], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)


# -------------------- Gaussian decoder p(y|z,X) -------------------- #
class CVAEDecoderGaussian(nn.Module):
    """
    Decoder p(y | z, X) = N(mu(z,X), diag(exp(logvar(z,X)))).
    Returns (mu, logvar) of shape (B, Dy).
    """

    def __init__(
        self,
        latent_dim: int,
        X_dim: int,
        y_dim: int,
        hidden: int = 128,
        min_logvar: float = -6.0,
        max_logvar: float = 4.0,
    ):
        super().__init__()
        self.y_dim = int(y_dim)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

        in_dim = latent_dim + X_dim
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
        self, z: torch.Tensor, X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(torch.cat([z, X], dim=1))
        mu = self.mu_head(h)  # (B, Dy)
        logvar = self.logvar_head(h)  # (B, Dy)
        logvar = torch.clamp(logvar, min=self.min_logvar, max=self.max_logvar)
        return mu, logvar

    @staticmethod
    def gaussian_nll(
        y: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean negative log-likelihood under N(mu, diag(exp(logvar))).
        """
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
        """One draw per row."""
        std = torch.exp(0.5 * logvar)
        # torch.randn_like may not accept `generator` on some versions -> use randn
        eps = torch.randn(
            std.shape,
            device=std.device,
            dtype=std.dtype,
            generator=generator,  # safe here
        )
        return mu + std * eps


# -------------------- CVAE (Gaussian) Estimator -------------------- #
class CVAEEstimator(ProbabilisticEstimator):
    """
    CVAE with a single Gaussian decoder (no MDN).

    Trains by minimizing: NLL(y | z, X) + β * KL[q(z|y,X) || p(z|X)]
      - Reconstruction: Gaussian NLL
      - KL: conditional prior alignment with warm-up and optional free-bits
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

        self._X_dim: int | None = None
        self._y_dim: int | None = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epochs = epochs
        self.batch_size = batch_size
        self.val_size = val_size
        self.random_state = random_state

    @property
    def type(self) -> str:
        # Add EstimatorTypeEnum.CVAE_GAUSS (or rename to your enum value)
        return getattr(EstimatorTypeEnum, "CVAE", EstimatorTypeEnum.CVAE).value

    # ---- data ---- #
    def _prepare_dataloaders(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ):
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
        """Fit the CVAE using the configured training hyperparameters."""
        super().fit(X, y)

        self._X_dim = int(X.shape[1])
        self._y_dim = int(y.shape[1])

        self._encoder = CVAEEncoder(
            y_dim=self._y_dim,
            X_dim=self._X_dim,
            latent_dim=self._latent_dim,
            hidden=self._hidden,
        ).to(self.device)

        self._decoder = CVAEDecoderGaussian(
            latent_dim=self._latent_dim,
            X_dim=self._X_dim,
            y_dim=self._y_dim,
            hidden=self._hidden,
            min_logvar=self._dec_min_lv,
            max_logvar=self._dec_max_lv,
        ).to(self.device)

        self._prior_net = PriorNet(
            cond_dim=self._X_dim,
            latent_dim=self._latent_dim,
            hidden=self._hidden,
            min_logvar=self._prior_min_lv,
            max_logvar=self._prior_max_lv,
        ).to(self.device)

        train_loader, val_loader = self._prepare_dataloaders(X, y)

        params = (
            list(self._encoder.parameters())
            + list(self._decoder.parameters())
            + list(self._prior_net.parameters())
        )
        opt = torch.optim.Adam(params, lr=self._learning_rate)

        self._training_history = TrainingHistory()

        def kl_gaussians(mu_q, logvar_q, mu_p, logvar_p):
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

            for Xb, yb in train_loader:
                Xb = Xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()

                # q(z|y,X)
                mu_q, logvar_q = self._encoder(yb, Xb)
                std_q = torch.exp(0.5 * logvar_q)
                z = mu_q + std_q * torch.randn_like(std_q)

                # p(z|X)
                mu_p, logvar_p = self._prior_net(Xb)

                # p(y|z,X) Gaussian
                mu_y, logvar_y = self._decoder(z, Xb)
                recon_nll = CVAEDecoderGaussian.gaussian_nll(yb, mu_y, logvar_y)
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
                for Xv, yv in val_loader:
                    Xv = Xv.to(self.device)
                    yv = yv.to(self.device)

                    mu_q_v, logvar_q_v = self._encoder(yv, Xv)
                    z_v = mu_q_v + torch.exp(0.5 * logvar_q_v) * torch.randn_like(
                        logvar_q_v
                    )
                    mu_p_v, logvar_p_v = self._prior_net(Xv)

                    mu_y_v, logvar_y_v = self._decoder(z_v, Xv)
                    recon_nll_v = CVAEDecoderGaussian.gaussian_nll(
                        yv, mu_y_v, logvar_y_v
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

    # ---- latent helper ---- #
    def _sample_z_given_X(
        self,
        X_tensor: torch.Tensor,
        n_samples: int,
        temperature: float,
        seed: int,
    ) -> torch.Tensor:
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
        )  # (n,S,latent)

    # ---- public API ---- #
    def sample(
        self,
        X: NDArray[np.float64],
        n_samples: int = 200,
        seed: int = 43,
        temperature: float = 1.0,
        use_prior: bool = True,
        max_outputs_per_chunk: int = 20_000,
    ) -> NDArray[np.float64]:
        """
        Draw `n_samples` IID samples per input from p(y|X):
          z ~ p(z|X) (or N(0,I) if use_prior=False), then y ~ N(mu(z,X), diag(exp(logvar))).
        """
        if self._decoder is None:
            raise RuntimeError("Model not fitted.")
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")

        self._decoder.eval()
        self._prior_net.eval()

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        n = X_t.shape[0]
        S = int(n_samples)
        Dy = int(self._y_dim)

        # latent z
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        if use_prior:
            z = self._sample_z_given_X(
                X_t, n_samples=S, temperature=temperature, seed=seed
            )  # (n,S,latent)
        else:
            z = torch.randn((n, S, self._latent_dim), device=self.device) * float(
                temperature
            )

        total = n * S
        z_flat = z.reshape(total, self._latent_dim)
        X_rep = X_t.unsqueeze(1).expand(n, S, self._X_dim).reshape(total, self._X_dim)

        # decode in chunks and draw one y per row
        results = []
        with torch.no_grad():
            if total <= max_outputs_per_chunk:
                mu, logvar = self._decoder(z_flat, X_rep)  # (T,Dy)
                y_flat = CVAEDecoderGaussian.sample(mu, logvar)  # (T,Dy)
            else:
                chunks = []
                for start in range(0, total, max_outputs_per_chunk):
                    end = min(total, start + max_outputs_per_chunk)
                    mu_c, lv_c = self._decoder(z_flat[start:end], X_rep[start:end])
                    y_c = CVAEDecoderGaussian.sample(mu_c, lv_c)
                    chunks.append(y_c)
                y_flat = torch.cat(chunks, dim=0)

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
        """One stochastic draw per input (wrapper over sample)."""
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
        use_prior: bool = True,
        max_outputs_per_chunk: int = 20_000,
    ) -> NDArray[np.float64]:
        """
        Conditional mean E[y|X].

        deterministic=True:
          Use a single z = μ_p(X), decode once, and return μ(z,X) (no observation sampling).
        deterministic=False:
          Monte-Carlo over z ~ p(z|X): compute μ(z_i,X) for i=1..S and average across i.
          (Integrates out observation noise analytically; no y sampling.)
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

        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        with torch.no_grad():
            if use_prior:
                mu_p, logvar_p = self._prior_net(X_t)
            else:
                mu_p = torch.zeros((n, self._latent_dim), device=self.device)
                logvar_p = torch.zeros_like(mu_p)
            std_p = torch.exp(0.5 * logvar_p)

            if deterministic:
                S = 1
                z = mu_p.unsqueeze(1)
            else:
                S = int(max(1, n_samples))
                eps = torch.randn((n, S, self._latent_dim), device=self.device)
                z = mu_p.unsqueeze(1) + float(temperature) * std_p.unsqueeze(1) * eps

            total = n * S
            z_flat = z.reshape(total, self._latent_dim)
            X_rep = (
                X_t.unsqueeze(1).expand(n, S, self._X_dim).reshape(total, self._X_dim)
            )

            # decode to (mu, logvar); take μ only (no component/obs sampling)
            if total <= max_outputs_per_chunk:
                mu_flat, _ = self._decoder(z_flat, X_rep)  # (T,Dy)
            else:
                mus = []
                for start in range(0, total, max_outputs_per_chunk):
                    end = min(total, start + max_outputs_per_chunk)
                    mu_c, _ = self._decoder(z_flat[start:end], X_rep[start:end])
                    mus.append(mu_c)
                mu_flat = torch.cat(mus, dim=0)

            mu = mu_flat.view(n, S, Dy)
            out = mu.mean(dim=1) if S > 1 else mu.squeeze(1)

        return out.cpu().numpy().astype(np.float64, copy=False)

    def predict_map(
        self,
        X: NDArray[np.float64],
        temperature: float = 1.0,
        seed: int = 43,
        use_prior: bool = True,
        max_outputs_per_chunk: int = 20_000,
    ) -> NDArray[np.float64]:
        """MAP estimate obtained by decoding the prior mean (deterministic path)."""

        return self.predict_mean(
            X,
            n_samples=1,
            seed=seed,
            temperature=temperature,
            deterministic=True,
            use_prior=use_prior,
            max_outputs_per_chunk=max_outputs_per_chunk,
        )

    def predict_median(
        self,
        X: NDArray[np.float64],
        n_samples: int = 501,
        seed: int = 43,
        temperature: float = 1.0,
        use_prior: bool = True,
        max_outputs_per_chunk: int = 20_000,
    ) -> NDArray[np.float64]:
        """Approximate posterior median by Monte-Carlo draws from p(y|X)."""

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
