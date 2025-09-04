from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from ....domain.model_management.interfaces.base_estimator import (
    ProbabilisticEstimator,
    TrainingHistory,  # <- uses the shared dataclass with .log() / .as_dict()
)


# -------------------- Conditional prior network -------------------- #
class PriorNet(nn.Module):
    """Simple conditional Gaussian prior p(z | cond). Outputs mu_p, logvar_p."""

    def __init__(self, cond_dim: int, latent_dim: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(cond_dim, hidden)
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)

    def forward(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(cond))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


# -------------------- Encoder / Decoder -------------------- #
class CVAEEncoder(nn.Module):
    """
    Encoder for CVAE: encodes decision `x_dec` and condition `x_cond` together
    and returns (mu, logvar) of q(z | decision, condition).
    """

    def __init__(self, x_dim: int, y_dim: int, latent_dim: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(x_dim + y_dim, 64)
        self.fc21 = nn.Linear(64, latent_dim)
        self.fc22 = nn.Linear(64, latent_dim)

    def forward(
        self, x_dec: torch.Tensor, x_cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(torch.cat([x_dec, x_cond], dim=1)))
        return self.fc21(h), self.fc22(h)


class CVAEDecoder(nn.Module):
    """Decoder module for the CVAE: maps (z, condition) -> reconstructed decision."""

    def __init__(self, latent_dim: int, y_dim: int, x_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + y_dim, 64)
        self.fc2 = nn.Linear(64, x_dim)

    @property
    def type(self) -> str:
        return "CVAE"

    def forward(self, z: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(torch.cat([z, x_cond], dim=1)))
        return self.fc2(h)


# -------------------- CVAE Estimator -------------------- #
class CVAEEstimator(ProbabilisticEstimator):
    """
    Conditional VAE estimator: models p(decision | objective) using CVAE.
      - fit(X, y): X = objectives (conditions), y = decisions (recon targets)
      - predict(X, n_samples): samples decisions given objectives
    """

    def __init__(self, latent_dim: int = 8, learning_rate: float = 1e-4):
        super().__init__()
        self._latent_dim = latent_dim
        self._learning_rate = learning_rate

        self._encoder: Optional[CVAEEncoder] = None
        self._decoder: Optional[CVAEDecoder] = None
        self._prior_net: Optional[PriorNet] = None

        self._y_dim: Optional[int] = None  # condition dim
        self._x_dim: Optional[int] = None  # decision dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def type(self) -> str:
        return "Conditional VAE"

    def _prepare_dataloaders(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        batch_size: int = 64,
        val_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[DataLoader, DataLoader]:
        """Split X/y into train/validation and return DataLoaders."""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=random_state
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

    # -------------------- training -------------------- #
    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64], **kwargs) -> None:
        """
        Train the CVAE. Uses internal train/val split.

        Optional kwargs:
          - epochs (int)
          - batch_size (int)
          - val_size (float)
          - random_state (int)
          - kl_weight (float): multiplier on KL term (default 1.0)
          - kl_warmup (int): warmup epochs for KL weight (default 0)
        """
        epochs = int(kwargs.get("epochs", 100))
        batch_size = int(kwargs.get("batch_size", 64))
        val_size = float(kwargs.get("val_size", 0.2))
        random_state = int(kwargs.get("random_state", 42))
        kl_weight = float(kwargs.get("kl_weight", 1.0))
        kl_warmup = int(kwargs.get("kl_warmup", 0))

        # dims (X = condition, y = decision)
        self._y_dim = int(X.shape[1])
        self._x_dim = int(y.shape[1])

        # modules
        self._encoder = CVAEEncoder(self._x_dim, self._y_dim, self._latent_dim).to(
            self.device
        )
        self._decoder = CVAEDecoder(self._latent_dim, self._y_dim, self._x_dim).to(
            self.device
        )
        self._prior_net = PriorNet(self._y_dim, self._latent_dim, hidden=64).to(
            self.device
        )

        # data
        train_loader, val_loader = self._prepare_dataloaders(
            X, y, batch_size=batch_size, val_size=val_size, random_state=random_state
        )
        n_train_samples = len(train_loader.dataset)
        n_val_samples = len(val_loader.dataset)

        # optimizer (joint)
        params = (
            list(self._encoder.parameters())
            + list(self._decoder.parameters())
            + list(self._prior_net.parameters())
        )
        optimizer = torch.optim.Adam(params, lr=self._learning_rate)

        # reset unified history
        self._training_history = TrainingHistory()

        for epoch in range(1, epochs + 1):
            # linear KL warmup in [0,1] multiplied by kl_weight
            kl_w = (
                min(1.0, epoch / max(1, kl_warmup)) if kl_warmup > 0 else 1.0
            ) * kl_weight

            # ---------------- TRAIN ----------------
            self._encoder.train()
            self._decoder.train()
            self._prior_net.train()
            train_recon_sum = 0.0
            train_kl_sum = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)  # condition
                y_batch = y_batch.to(self.device)  # decision

                optimizer.zero_grad()

                # q(z | y, X)
                mu_q, logvar_q = self._encoder(y_batch, X_batch)
                std_q = torch.exp(0.5 * logvar_q)
                z = mu_q + std_q * torch.randn_like(std_q)

                # p(z | X)
                mu_p, logvar_p = self._prior_net(X_batch)

                # recon in decision space
                recon = self._decoder(z, X_batch)
                recon_loss = F.mse_loss(recon, y_batch, reduction="sum")

                # KL(q||p) for diagonal Gaussians
                kl = 0.5 * torch.sum(
                    (logvar_p - logvar_q)
                    + (torch.exp(logvar_q) + (mu_q - mu_p).pow(2))
                    / (torch.exp(logvar_p) + 1e-8)
                    - 1.0
                )

                loss = recon_loss + kl_w * kl
                loss.backward()
                optimizer.step()

                train_recon_sum += float(recon_loss.item())
                train_kl_sum += float(kl.item())

            avg_recon = train_recon_sum / max(1, n_train_samples)
            avg_kl = train_kl_sum / max(1, n_train_samples)
            avg_total = avg_recon + kl_w * avg_kl

            # ---------------- VALIDATION ----------------
            self._encoder.eval()
            self._decoder.eval()
            self._prior_net.eval()
            val_recon_sum = 0.0
            val_kl_sum = 0.0
            with torch.no_grad():
                for Xv, yv in val_loader:
                    Xv = Xv.to(self.device)
                    yv = yv.to(self.device)

                    mu_q_v, logvar_q_v = self._encoder(yv, Xv)
                    std_q_v = torch.exp(0.5 * logvar_q_v)
                    zv = mu_q_v + std_q_v * torch.randn_like(std_q_v)

                    mu_p_v, logvar_p_v = self._prior_net(Xv)
                    recon_v = self._decoder(zv, Xv)

                    recon_loss_v = F.mse_loss(recon_v, yv, reduction="sum")
                    kl_v = 0.5 * torch.sum(
                        (logvar_p_v - logvar_q_v)
                        + (torch.exp(logvar_q_v) + (mu_q_v - mu_p_v).pow(2))
                        / (torch.exp(logvar_p_v) + 1e-8)
                        - 1.0
                    )

                    val_recon_sum += float(recon_loss_v.item())
                    val_kl_sum += float(kl_v.item())

            avg_val_recon = val_recon_sum / max(1, n_val_samples)
            avg_val_kl = val_kl_sum / max(1, n_val_samples)
            avg_val_total = avg_val_recon + kl_w * avg_val_kl

            # ---------------- LOG (unified) ----------------
            self._training_history.log(
                epoch=epoch,
                train_loss=avg_total,
                val_loss=avg_val_total,
                train_recon=avg_recon,
                train_kl=avg_kl,
                val_recon=avg_val_recon,
                val_kl=avg_val_kl,
            )

    # -------------------- sampling utilities -------------------- #
    def _sample_z_given_y(
        self,
        cond_tensor: torch.Tensor,
        n_samples: int,
        temperature: float,
        seed: Optional[int],
    ) -> torch.Tensor:
        """
        Returns z tensor with shape (n, n_samples, latent)
        using prior_net(cond). If prior_net is missing, draw std normal.
        """
        n = cond_tensor.shape[0]
        if self._prior_net is not None:
            mu_p, logvar_p = self._prior_net(cond_tensor)  # (n, latent)
            std_p = torch.exp(0.5 * logvar_p)
            if seed is not None:
                gen = torch.Generator(device=self.device)
                gen.manual_seed(int(seed))
                eps = torch.randn(
                    (n, n_samples, self._latent_dim), generator=gen, device=self.device
                )
            else:
                eps = torch.randn((n, n_samples, self._latent_dim), device=self.device)
            return mu_p.unsqueeze(1) + temperature * std_p.unsqueeze(1) * eps

        if seed is not None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(seed))
            return torch.randn(
                (n, n_samples, self._latent_dim), generator=gen, device=self.device
            ) * float(temperature)
        return torch.randn(
            (n, n_samples, self._latent_dim), device=self.device
        ) * float(temperature)

    # -------------------- predict & convenience methods -------------------- #

    def predict(
        self,
        X: np.ndarray,
        n_samples: int = 1,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        use_prior: bool = True,
        *,
        max_outputs_per_chunk: int = 20000,
    ) -> np.ndarray:
        """
        One call wrapper over `sample()`.
        - If n_samples == 1 returns (n, x_dim)
        - Else returns (n, n_samples, x_dim)
        """
        return self.sample(
            X,
            n_samples=n_samples,
            seed=seed,
            temperature=temperature,
            use_prior=use_prior,
            max_outputs_per_chunk=max_outputs_per_chunk,
        )

    def sample(
        self,
        X: NDArray[np.float64],
        n_samples: int = 200,
        seed: Optional[int] = None,
        temperature: float = 1.0,
        use_prior: bool = True,
        max_outputs_per_chunk: int = 20000,
    ) -> NDArray[np.float64]:
        """
        Vectorized sampling from p(decision | objective).
          - returns (n, x_dim) if n_samples == 1
          - returns (n, n_samples, x_dim) otherwise
        """
        if self._decoder is None:
            raise RuntimeError("Model not fitted.")

        self._decoder.eval()
        if self._prior_net is not None:
            self._prior_net.eval()

        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        n = X_tensor.shape[0]
        n_samples = int(n_samples)

        if use_prior:
            z = self._sample_z_given_y(
                X_tensor, n_samples=n_samples, temperature=temperature, seed=seed
            )
        else:
            if seed is not None:
                gen = torch.Generator(device=self.device)
                gen.manual_seed(int(seed))
                z = torch.randn(
                    (n, n_samples, self._latent_dim), generator=gen, device=self.device
                ) * float(temperature)
            else:
                z = torch.randn(
                    (n, n_samples, self._latent_dim), device=self.device
                ) * float(temperature)

        total = n * n_samples
        x_dim = int(self._x_dim)
        latent = int(self._latent_dim)

        if total <= max_outputs_per_chunk:
            z_flat = z.view(total, latent)
            X_rep = (
                X_tensor.unsqueeze(1)
                .expand(n, n_samples, X_tensor.shape[1])
                .contiguous()
                .view(total, X_tensor.shape[1])
            )
            with torch.no_grad():
                out_flat = self._decoder(z_flat, X_rep)  # (total, x_dim)
            out = out_flat.view(n, n_samples, x_dim).cpu().numpy()
            return out[:, 0, :] if n_samples == 1 else out

        # chunked decode
        results = []
        chunk_size = int(max_outputs_per_chunk)
        z_flat_all = z.view(total, latent)
        for s in range(0, total, chunk_size):
            e = min(s + chunk_size, total)
            z_chunk = z_flat_all[s:e]
            idx_flat = torch.arange(s, e, device=self.device)
            cond_idx = (idx_flat // n_samples).long()  # indices in 0..n-1
            X_chunk = X_tensor[cond_idx]
            with torch.no_grad():
                out_chunk = self._decoder(z_chunk, X_chunk)
            results.append(out_chunk.cpu())

        out_all = torch.cat(results, dim=0)
        out_np = out_all.numpy().reshape(n, n_samples, x_dim)
        return out_np[:, 0, :] if n_samples == 1 else out_np

    def predict_mean(
        self,
        X: NDArray[np.float64],
        n_samples: int = 200,
        seed: Optional[int] = None,
        temperature: float = 1.0,
        use_prior: bool = True,
        max_outputs_per_chunk: int = 20000,
    ) -> NDArray[np.float64]:
        """Estimate conditional mean by sampling & averaging. Returns (n, x_dim)."""
        s = self.sample(
            X,
            n_samples=n_samples,
            seed=seed,
            temperature=temperature,
            use_prior=use_prior,
            max_outputs_per_chunk=max_outputs_per_chunk,
        )
        return s if s.ndim == 2 else s.mean(axis=1)

    def predict_quantiles(
        self,
        X: NDArray[np.float64],
        n_samples: int = 500,
        qs: Sequence[float] = (0.05, 0.95),
        seed: Optional[int] = None,
        temperature: float = 1.0,
        use_prior: bool = True,
        max_outputs_per_chunk: int = 20000,
    ) -> NDArray[np.float64]:
        """Return (n, len(qs), x_dim) Monte-Carlo quantiles."""
        s = self.sample(
            X,
            n_samples=n_samples,
            seed=seed,
            temperature=temperature,
            use_prior=use_prior,
            max_outputs_per_chunk=max_outputs_per_chunk,
        )
        if s.ndim == 2:
            s = s[:, None, :]
        percents = [100.0 * float(q) for q in qs]
        q_arr = np.percentile(s, percents, axis=1)  # (len(qs), n, x_dim)
        return np.transpose(q_arr, (1, 0, 2))
