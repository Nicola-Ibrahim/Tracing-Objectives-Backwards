"""
Invertible Neural Network (INN) / Normalizing Flow Estimator for Inverse Problems.

This module implements a conditional normalizing flow using RealNVP-style affine coupling
layers to model posterior distributions p(x|y) where multiple solutions x can explain
the same observation y.

Architecture:
    - Conditional flow with affine coupling layers
    - Alternating split patterns for invertibility
    - Batch normalization for training stability
    - Base distribution: Standard Gaussian N(0, I)

Reference:
    Dinh et al. "Density estimation using Real NVP" (2017)
"""

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .....domain.modeling.enums.estimator_type import EstimatorTypeEnum
from .....domain.modeling.interfaces.base_estimator import ProbabilisticEstimator
from .....domain.modeling.value_objects.estimator_params import (
    INNEstimatorParams,
    INNOptimizerEnum,
)

# ======================================================================
# Coupling Layer Components
# ======================================================================


class AffineCouplingLayer(nn.Module):
    """
    RealNVP-style affine coupling layer with conditional input.

    Splits input into two parts [x1, x2]:
    - x1 passes through unchanged
    - x2 is transformed: x2' = x2 * exp(s(x1, cond)) + t(x1, cond)

    where s (scale) and t (translation) are learned neural networks.
    """

    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        hidden_dim: int = 128,
        mask_type: str = "even",
    ):
        """
        Args:
            input_dim: Dimension of input x
            cond_dim: Dimension of conditioning variable y
            hidden_dim: Hidden layer size for s and t networks
            mask_type: 'even' or 'odd' - determines which half is transformed
        """
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.mask_type = mask_type

        # Create mask: 1 for unchanged part, 0 for transformed part
        self.register_buffer("mask", self._create_mask(input_dim, mask_type))

        # Number of dimensions to transform
        split_dim = input_dim // 2
        unchanged_dim = input_dim - split_dim

        # Networks for scale (s) and translation (t)
        # Input: unchanged part + conditioning
        net_input_dim = unchanged_dim + cond_dim

        # Shared backbone
        self.net = nn.Sequential(
            nn.Linear(net_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Scale network (s) - output log-scale for numerical stability
        self.scale_net = nn.Linear(hidden_dim, split_dim)
        # Initialize to near-zero for stable training
        nn.init.zeros_(self.scale_net.weight)
        nn.init.zeros_(self.scale_net.bias)

        # Translation network (t)
        self.translation_net = nn.Linear(hidden_dim, split_dim)
        nn.init.zeros_(self.translation_net.weight)
        nn.init.zeros_(self.translation_net.bias)

    def _create_mask(self, dim: int, mask_type: str) -> torch.Tensor:
        """Create binary mask for splitting input."""
        mask = torch.zeros(dim)
        if mask_type == "even":
            mask[::2] = 1  # Keep even indices unchanged
        else:  # odd
            mask[1::2] = 1  # Keep odd indices unchanged
        return mask

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, reverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward or inverse transformation.

        Args:
            x: Input tensor (batch, input_dim)
            cond: Conditioning tensor (batch, cond_dim)
            reverse: If True, compute inverse transformation

        Returns:
            transformed_x: Transformed tensor
            log_det_jacobian: Log determinant of Jacobian
        """
        # Split input
        x1 = x * self.mask  # Unchanged part
        x2 = x * (1 - self.mask)  # Part to transform

        # Compute scale and translation from unchanged part and conditioning
        x1_unmasked = x1[:, self.mask.bool()]
        net_input = torch.cat([x1_unmasked, cond], dim=1)
        h = self.net(net_input)

        # Get scale (log-space) and translation
        log_s = self.scale_net(h)
        log_s = torch.tanh(log_s)  # Bound scale for stability
        t = self.translation_net(h)

        if not reverse:
            # Forward: x2' = x2 * exp(s) + t
            x2_unmasked = x2[:, (~self.mask.bool())]
            x2_transformed = x2_unmasked * torch.exp(log_s) + t

            # Reconstruct full output
            out = x1.clone()
            out[:, ~self.mask.bool()] = x2_transformed

            # Log determinant is sum of log scales
            log_det = log_s.sum(dim=1)
        else:
            # Inverse: x2 = (x2' - t) / exp(s) = (x2' - t) * exp(-s)
            x2_unmasked = x2[:, (~self.mask.bool())]
            x2_original = (x2_unmasked - t) * torch.exp(-log_s)

            # Reconstruct full output
            out = x1.clone()
            out[:, ~self.mask.bool()] = x2_original

            # Log determinant for inverse is negative
            log_det = -log_s.sum(dim=1)

        return out, log_det


# ======================================================================
# Normalizing Flow Model
# ======================================================================


class ConditionalNormalizingFlow(nn.Module):
    """
    Complete normalizing flow model with multiple coupling layers.

    Transforms between:
    - Base distribution: z ~ N(0, I)
    - Data distribution: x | y

    Forward: z -> x (sampling)
    Inverse: x -> z (likelihood evaluation)
    """

    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        num_coupling_layers: int = 6,
        hidden_dim: int = 128,
        use_batch_norm: bool = True,
    ):
        """
        Args:
            input_dim: Dimension of data x
            cond_dim: Dimension of conditioning y
            num_coupling_layers: Number of coupling transformations
            hidden_dim: Hidden layer size in coupling layers
            use_batch_norm: Whether to use batch normalization between layers
        """
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.num_coupling_layers = num_coupling_layers

        # Build flow layers
        self.coupling_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None

        for i in range(num_coupling_layers):
            # Alternate mask pattern for each layer
            mask_type = "even" if i % 2 == 0 else "odd"

            coupling = AffineCouplingLayer(
                input_dim=input_dim,
                cond_dim=cond_dim,
                hidden_dim=hidden_dim,
                mask_type=mask_type,
            )
            self.coupling_layers.append(coupling)

            if use_batch_norm:
                # Batch norm for stability (in data space, not conditioning)
                self.batch_norms.append(nn.BatchNorm1d(input_dim, affine=True))

    def forward(
        self, z: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward transformation: z -> x (for sampling).

        Args:
            z: Latent samples from N(0, I) (batch, input_dim)
            cond: Conditioning inputs (batch, cond_dim)

        Returns:
            x: Transformed samples
            log_det: Log determinant of Jacobian (batch,)
        """
        x = z
        log_det_sum = torch.zeros(z.shape[0], device=z.device)

        for i, coupling in enumerate(self.coupling_layers):
            x, log_det = coupling(x, cond, reverse=False)
            log_det_sum += log_det

            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
                # Batch norm Jacobian determinant
                # log|det(BN)| = sum(log|gamma|) where gamma are the scale parameters
                log_det_bn = torch.log(torch.abs(self.batch_norms[i].weight)).sum()
                log_det_sum += log_det_bn

        return x, log_det_sum

    def inverse(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse transformation: x -> z (for likelihood evaluation).

        Args:
            x: Data samples (batch, input_dim)
            cond: Conditioning inputs (batch, cond_dim)

        Returns:
            z: Latent codes
            log_det: Log determinant of Jacobian (batch,)
        """
        z = x
        log_det_sum = torch.zeros(x.shape[0], device=x.device)

        # Apply layers in reverse order
        for i in reversed(range(len(self.coupling_layers))):
            if self.batch_norms is not None:
                # Inverse batch norm
                z = (z - self.batch_norms[i].bias) / self.batch_norms[i].weight
                # Determinant for inverse BN
                log_det_bn = -torch.log(torch.abs(self.batch_norms[i].weight)).sum()
                log_det_sum += log_det_bn

            z, log_det = self.coupling_layers[i](z, cond, reverse=True)
            log_det_sum += log_det

        return z, log_det_sum

    def log_prob(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(x|cond) using change of variables.

        log p(x|cond) = log p(z) + log|det J_f^{-1}|
        where z = f^{-1}(x|cond) and p(z) = N(0, I)
        """
        z, log_det_inv = self.inverse(x, cond)

        # Log prob of z under standard Gaussian
        log_pz = -0.5 * (z**2 + np.log(2 * np.pi)).sum(dim=1)

        # Change of variables formula
        log_px = log_pz + log_det_inv

        return log_px


# ======================================================================
# INN Estimator
# ======================================================================


class INNEstimator(ProbabilisticEstimator):
    """
    Invertible Neural Network (Normalizing Flow) estimator for inverse problems.

    Models the posterior distribution p(x|y) using a conditional normalizing flow,
    enabling diverse sampling of multiple solutions x that could explain observation y.

    Public API:
        - fit(X, y): Train the flow on paired (condition, target) data
        - sample(X, n_samples, ...): Draw samples from p(y|X)
        - get_loss_history(): Return training metrics
    """

    def __init__(self, params: INNEstimatorParams):
        super().__init__()
        self.params = params

        # Architecture
        self._num_coupling_layers = params.num_coupling_layers
        self._hidden_dim = params.hidden_dim
        self._use_batch_norm = params.use_batch_norm

        # Optimization
        self._learning_rate = params.learning_rate
        self._optimizer_name = params.optimizer_name
        self._weight_decay = params.weight_decay
        self.clip_grad_norm = params.clip_grad_norm

        # Training
        self.epochs = params.epochs
        self.batch_size = params.batch_size
        self.val_size = params.val_size
        self.seed = params.seed
        self._verbose = params.verbose

        # Scheduling & early stopping
        self.lr_scheduler = params.lr_scheduler
        self.lr_decay_factor = params.lr_decay_factor
        self.lr_patience = params.lr_patience
        self.early_stopping_patience = params.early_stopping_patience

        # Model components (initialized in fit)
        self._flow: ConditionalNormalizingFlow | None = None
        self._best_model_state: dict | None = None

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def type(self) -> str:
        return getattr(EstimatorTypeEnum, "INN", EstimatorTypeEnum.INN).value

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> None:
        """
        Train the normalizing flow on paired data.

        Args:
            X: Conditioning inputs (n_samples, cond_dim) - typically objectives
            y: Target outputs (n_samples, output_dim) - typically decisions
        """
        # Base validation
        super().fit(X, y)

        # Set random seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        cond_dim = X.shape[1]
        output_dim = y.shape[1]

        # Initialize flow
        self._flow = ConditionalNormalizingFlow(
            input_dim=output_dim,
            cond_dim=cond_dim,
            num_coupling_layers=self._num_coupling_layers,
            hidden_dim=self._hidden_dim,
            use_batch_norm=self._use_batch_norm,
        ).to(self._device)

        # Prepare data
        train_loader, val_loader = self._prepare_dataloaders(X, y)

        # Optimizer
        optimizer = self._get_optimizer()

        # Learning rate scheduler
        scheduler = None
        if self.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.lr_decay_factor,
                patience=self.lr_patience,
            )

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            # Train
            train_loss = self._train_epoch(train_loader, optimizer)

            # Validate
            val_loss = self._validate_epoch(val_loader)

            # Log history
            self._training_history.log(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
            )

            # Learning rate scheduling
            if scheduler is not None:
                scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._best_model_state = self._flow.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                if self._verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

            # Verbose logging
            if self._verbose and epoch % 10 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch:3d}/{self.epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"LR: {current_lr:.2e}"
                )

        # Load best model
        if self._best_model_state is not None:
            self._flow.load_state_dict(self._best_model_state)

        self._flow.eval()

    def _train_epoch(
        self, train_loader: DataLoader, optimizer: torch.optim.Optimizer
    ) -> float:
        """Run one training epoch."""
        self._flow.train()
        total_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self._device)
            batch_y = batch_y.to(self._device)

            optimizer.zero_grad()

            # Compute negative log-likelihood
            log_prob = self._flow.log_prob(batch_y, batch_X)
            loss = -log_prob.mean()

            loss.backward()

            # Gradient clipping
            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self._flow.parameters(), self.clip_grad_norm
                )

            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Run validation epoch."""
        self._flow.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self._device)
                batch_y = batch_y.to(self._device)

                log_prob = self._flow.log_prob(batch_y, batch_X)
                loss = -log_prob.mean()
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def sample(
        self,
        X: npt.NDArray[np.float64],
        n_samples: int = 50,
        seed: int = 42,
        temperature: float = 1.0,
        **kwargs,
    ) -> npt.NDArray[np.float64]:
        """
        Draw samples from p(y|X) using the flow.

        Args:
            X: Conditioning inputs (n, cond_dim)
            n_samples: Number of samples per input
            seed: Random seed
            temperature: Temperature for sampling (higher = more diverse)

        Returns:
            samples: Shape (n, n_samples, output_dim) if n_samples > 1
                    Shape (n, output_dim) if n_samples == 1
        """
        if self._flow is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")

        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self._flow.eval()

        X_tensor = torch.tensor(X, dtype=torch.float32, device=self._device)
        n = X_tensor.shape[0]
        output_dim = self._flow.input_dim

        with torch.no_grad():
            # Sample from base distribution
            z = torch.randn(n, n_samples, output_dim, device=self._device) * np.sqrt(
                temperature
            )

            # Transform through flow
            samples_list = []
            for i in range(n_samples):
                z_i = z[:, i, :]  # (n, output_dim)
                cond_i = X_tensor  # (n, cond_dim)
                x_i, _ = self._flow.forward(z_i, cond_i)
                samples_list.append(x_i.unsqueeze(1))

            samples_3d = torch.cat(samples_list, dim=1)  # (n, n_samples, output_dim)

        samples_np = samples_3d.cpu().numpy().astype(np.float64, copy=False)

        if n_samples == 1:
            return samples_np[:, 0, :]  # (n, output_dim)
        else:
            return samples_np  # (n, n_samples, output_dim)

    # -------------------- Helper Methods --------------------

    def _prepare_dataloaders(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[DataLoader, DataLoader]:
        """Split data and create DataLoaders."""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_size, random_state=self.seed
        )

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader

    def _get_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        if self._optimizer_name == INNOptimizerEnum.ADAM:
            return torch.optim.Adam(
                self._flow.parameters(),
                lr=self._learning_rate,
                weight_decay=self._weight_decay,
            )
        elif self._optimizer_name == INNOptimizerEnum.ADAMW:
            return torch.optim.AdamW(
                self._flow.parameters(),
                lr=self._learning_rate,
                weight_decay=self._weight_decay,
            )
        elif self._optimizer_name == INNOptimizerEnum.SGD:
            return torch.optim.SGD(
                self._flow.parameters(),
                lr=self._learning_rate,
                weight_decay=self._weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self._optimizer_name}")

    # -------------------- Checkpoint Serialization --------------------

    def to_checkpoint(self) -> dict:
        """
        Serialize INN/Flow model state to JSON-serializable checkpoint.

        Returns:
            dict: Checkpoint containing flow weights and training state
        """
        if self._flow is None:
            raise RuntimeError("Cannot checkpoint unfitted model. Call fit() first.")

        # Keep tensors for safetensors serialization (namespaced keys)
        flow_state = {
            f"flow.{k}": v.detach().cpu() for k, v in self._flow.state_dict().items()
        }

        checkpoint = {
            "model_state": flow_state,
            "input_dim": int(self._flow.input_dim),
            "cond_dim": int(self._flow.cond_dim),
            "training_history": self._training_history.as_dict(),
        }

        return checkpoint

    @classmethod
    def from_checkpoint(cls, parameters: dict) -> "INNEstimator":
        """
        Reconstruct trained INN estimator from parameters.

        Args:
            parameters: Full parameters dict containing hyperparameters and model state

        Returns:
            INNEstimator: Fully initialized trained estimator
        """
        # Parse enum parameters from strings
        checkpoint_fields = {"model_state", "input_dim", "cond_dim", "training_history"}
        parsed_params = {}
        for key, value in parameters.items():
            if key == "optimizer_name":
                parsed_params[key] = (
                    INNOptimizerEnum(value) if isinstance(value, str) else value
                )
            elif key not in checkpoint_fields and key not in [
                "type",
                "mapping_direction",
            ]:  # Skip metadata fields
                parsed_params[key] = value

        estimator_params = INNEstimatorParams(**parsed_params)
        estimator = cls(estimator_params)

        # Restore dimensions
        input_dim = parameters["input_dim"]
        cond_dim = parameters["cond_dim"]
        estimator._X_dim = cond_dim  # BaseEstimator convention
        estimator._y_dim = input_dim

        # Rebuild flow architecture
        estimator._flow = ConditionalNormalizingFlow(
            input_dim=input_dim,
            cond_dim=cond_dim,
            num_coupling_layers=estimator._num_coupling_layers,
            hidden_dim=estimator._hidden_dim,
            use_batch_norm=estimator._use_batch_norm,
        ).to(estimator._device)

        # Load weights from checkpoint
        flow_state = {}
        for key, value in parameters["model_state"].items():
            if key.startswith("flow."):
                flow_state[key[len("flow.") :]] = (
                    value.to(estimator._device)
                    if torch.is_tensor(value)
                    else torch.tensor(value, device=estimator._device)
                )

        estimator._flow.load_state_dict(flow_state)
        estimator._flow.eval()

        # Restore training history
        if "training_history" in parameters:
            hist = parameters["training_history"]
            estimator._training_history.epochs = hist.get("epochs", [])
            estimator._training_history.train_loss = hist.get("train_loss", [])
            estimator._training_history.val_loss = hist.get("val_loss", [])
            for key, value in hist.items():
                if key not in ["epochs", "train_loss", "val_loss"]:
                    estimator._training_history.extras[key] = value

        return estimator
