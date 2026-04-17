from typing import Any, Self

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler

from ..domain.enums.transform_type import TransformTypeEnum
from ..domain.interfaces.base_transform import BaseTransformer


class MinMaxScalerNormalizer(BaseTransformer):
    """
    Concrete implementation using scikit-learn's MinMaxScaler.
    """

    def __init__(self, feature_range: tuple[float, float] = (0, 1)):
        self.feature_range = feature_range
        self.scaler = MinMaxScaler(feature_range=feature_range)

    @property
    def config(self) -> dict[str, Any]:
        return {
            "type": TransformTypeEnum.MIN_MAX.value,
            "params": {"feature_range": self.feature_range},
        }

    def get_fitted_state(self) -> dict[str, Any]:
        return {
            "scale_": self.scaler.scale_,
            "min_": self.scaler.min_,
            "data_min_": self.scaler.data_min_,
            "data_max_": self.scaler.data_max_,
            "data_range_": self.scaler.data_range_,
        }

    @classmethod
    def from_checkpoint(
        cls, config: dict[str, Any], state: dict[str, Any]
    ) -> "MinMaxScalerNormalizer":
        params = config.get("params", {})
        instance = cls(**params)
        for k, v in state.items():
            setattr(instance.scaler, k, v)
        return instance

    def fit(self, X: NDArray[np.floating], y: NDArray | None = None) -> Self:
        """
        Fits the scaler to the data.
        Args:
            data: Input data array (n_samples, n_features)
            y: Optional target values (ignored)
        Returns:
            The fitted scaler instance (self)
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.scaler.fit(X)
        return self

    def transform(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Transforms data using the fitted scaler."""
        if X.ndim == 1:
            X = X.reshape(1, -1)

        return self.scaler.transform(X)

    def inverse_transform(self, X_norm: NDArray[np.floating]) -> NDArray[np.floating]:
        """Inverse transforms data to the original scale."""
        if X_norm.ndim == 1:
            X_norm = X_norm.reshape(-1, 1)
        return self.scaler.inverse_transform(X_norm)


class HypercubeNormalizer(BaseTransformer):
    """
    Normalizes data to a specified range (default [0, 1]) feature-wise.
    Useful for scaling features to a common range without distorting differences.

    Args:
        feature_range: Desired range of transformed data (default: (0, 1))
        clip: Whether to clip transformed values to feature_range
    """

    def __init__(self, feature_range: tuple[float, float] = (0, 1), clip: bool = False):
        self.feature_range = feature_range
        self.clip = clip
        # Will store min and max values for each feature during fitting
        self.min_vals = None
        self.max_vals = None
        # Will store the range (max-min) for each feature
        self.ranges = None

    @property
    def config(self) -> dict[str, Any]:
        return {
            "type": TransformTypeEnum.HYPERCUBE.value,
            "params": {"feature_range": self.feature_range, "clip": self.clip},
        }

    def get_fitted_state(self) -> dict[str, Any]:
        return {
            "min_vals": self.min_vals,
            "max_vals": self.max_vals,
            "ranges": self.ranges,
        }

    @classmethod
    def from_checkpoint(
        cls, config: dict[str, Any], state: dict[str, Any]
    ) -> "HypercubeNormalizer":
        params = config.get("params", {})
        instance = cls(**params)
        instance.min_vals = state.get("min_vals")
        instance.max_vals = state.get("max_vals")
        instance.ranges = state.get("ranges")
        return instance

    def fit(self, X: NDArray[np.float64], y: NDArray | None = None) -> Self:
        """
        Compute minimum, maximum, and range for each feature/column.
        Handles constant features by setting range to 1.0 to avoid division by zero.
        """
        # Calculate min and max for each feature
        self.min_vals = np.min(X, axis=0)
        self.max_vals = np.max(X, axis=0)

        # Calculate ranges, handling constant features (where max == min)
        self.ranges = np.where(
            self.max_vals > self.min_vals,
            self.max_vals - self.min_vals,
            1.0,  # Default range for constant features
        )
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Scale data to [0, 1] range and optionally clip values"""
        if self.min_vals is None or self.max_vals is None:
            raise RuntimeError("Normalizer has not been fitted")

        # Normalize to [0, 1] range
        X_norm = (X - self.min_vals) / self.ranges

        # Scale to desired feature range
        min_target, max_target = self.feature_range
        X_norm = X_norm * (max_target - min_target) + min_target

        # Clip values if requested
        if self.clip:
            np.clip(X_norm, min_target, max_target, out=X_norm)

        return X_norm

    def inverse_transform(self, X_norm: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reverse the normalization to original scale"""
        if self.min_vals is None or self.max_vals is None:
            raise RuntimeError("Normalizer has not been fitted")

        min_target, max_target = self.feature_range
        # First scale back from feature_range to [0, 1]
        X_scaled = (X_norm - min_target) / (max_target - min_target)
        # Then scale back to original range
        return X_scaled * self.ranges + self.min_vals


class StandardNormalizer(BaseTransformer):
    """
    Standard normalization (z-score) transforms features to have zero mean
    and unit variance. This is useful for algorithms that assume features
    are centered and scaled.

    Args:
        with_mean: Whether to center the data (subtract mean)
        with_std: Whether to scale to unit variance (divide by std)
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean = None  # Will store feature means
        self.std = None  # Will store feature standard deviations

    @property
    def config(self) -> dict[str, Any]:
        return {
            "type": TransformTypeEnum.STANDARD.value,
            "params": {"with_mean": self.with_mean, "with_std": self.with_std},
        }

    def get_fitted_state(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "std": self.std,
        }

    @classmethod
    def from_checkpoint(
        cls, config: dict[str, Any], state: dict[str, Any]
    ) -> "StandardNormalizer":
        params = config.get("params", {})
        instance = cls(**params)
        instance.mean = state.get("mean")
        instance.std = state.get("std")
        return instance

    def fit(self, X: NDArray[np.float64], y: NDArray | None = None) -> Self:
        """Compute mean and standard deviation for each feature"""
        if self.with_mean:
            self.mean = np.mean(X, axis=0)
        else:
            # If not centering, use zero mean
            self.mean = 0.0

        if self.with_std:
            self.std = np.std(X, axis=0)
            # Handle constant columns by setting std to 1.0
            self.std[self.std < 1e-8] = 1.0
        else:
            # If not scaling, use unit std
            self.std = 1.0

        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply z-score normalization: (X - mean) / std"""
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer has not been fitted")
        return (X - self.mean) / self.std

    def inverse_transform(self, X_norm: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reverse z-score normalization: X_norm * std + mean"""
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer has not been fitted")
        return X_norm * self.std + self.mean


class UnitVectorNormalizer(BaseTransformer):
    """
    Normalizes vectors to unit length using L2 normalization.
    This is useful when the direction of the vector matters more than its magnitude.

    Args:
        axis: 0 to normalize columns (features), 1 to normalize rows (samples)
    """

    def __init__(self, axis: int = 1):
        self.axis = axis
        self.norm_values = None  # Will store L2 norms

    @property
    def config(self) -> dict[str, Any]:
        return {
            "type": TransformTypeEnum.UNIT_VECTOR.value,
            "params": {"axis": self.axis},
        }

    def get_fitted_state(self) -> dict[str, Any]:
        return {
            "norm_values": self.norm_values,
        }

    @classmethod
    def from_checkpoint(
        cls, config: dict[str, Any], state: dict[str, Any]
    ) -> "UnitVectorNormalizer":
        params = config.get("params", {})
        instance = cls(**params)
        instance.norm_values = state.get("norm_values")
        return instance

    def fit(self, X: NDArray[np.float64], y: NDArray | None = None) -> Self:
        """
        Compute L2 norms for each vector (row or column).
        Stores norms to use in transformation.
        """
        # Compute L2 norms along specified axis
        self.norm_values = np.linalg.norm(X, axis=self.axis)
        # Replace zero norms with 1.0 to prevent division by zero
        self.norm_values[self.norm_values == 0] = 1.0
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalize vectors to unit length"""
        if self.norm_values is None:
            raise RuntimeError("Normalizer has not been fitted")

        if self.axis == 1:
            # Normalize each row (sample) by its L2 norm
            return X / self.norm_values[:, np.newaxis]
        elif self.axis == 0:
            # Normalize each column (feature) by its L2 norm
            return X / self.norm_values
        else:
            raise ValueError(f"Unsupported axis: {self.axis}")

    def inverse_transform(self, X_norm: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reverse the normalization by scaling back with original norms"""
        if self.norm_values is None:
            raise RuntimeError("Normalizer has not been fitted")

        if self.axis == 1:
            return X_norm * self.norm_values[:, np.newaxis]
        elif self.axis == 0:
            return X_norm * self.norm_values
        else:
            raise ValueError(f"Unsupported axis: {self.axis}")


class LogNormalizer(BaseTransformer):
    """
    Applies logarithmic transformation to data.
    Useful for data with exponential distributions or large ranges.

    Args:
        offset: Value to add before taking log (to handle zeros)
        base: Logarithm base (e for natural log, 10 for common log)
    """

    def __init__(self, offset: float = 1.0, base: float = np.e):
        self.offset = offset
        self.base = base

    @property
    def config(self) -> dict[str, Any]:
        return {
            "type": TransformTypeEnum.LOG.value,
            "params": {"offset": self.offset, "base": self.base},
        }

    def get_fitted_state(self) -> dict[str, Any]:
        return {}

    @classmethod
    def from_checkpoint(
        cls, config: dict[str, Any], state: dict[str, Any]
    ) -> "LogNormalizer":
        params = config.get("params", {})
        return cls(**params)

    def fit(self, X: NDArray[np.float64], y: NDArray | None = None) -> Self:
        """No Args to learn for log transformation"""
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply log transformation: log_base(X + offset)"""
        X_shifted = X + self.offset
        if self.base == np.e:
            return np.log(X_shifted)  # Natural log
        else:
            return np.log(X_shifted) / np.log(self.base)  # General log

    def fit_transform(
        self, X: NDArray[np.float64], y: NDArray | None = None
    ) -> NDArray[np.float64]:
        """Fit to data, then transform it"""
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X_norm: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reverse log transformation: base^X_norm - offset"""
        if self.base == np.e:
            return np.exp(X_norm) - self.offset
        else:
            return np.power(self.base, X_norm) - self.offset
