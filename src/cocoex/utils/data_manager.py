from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class ParetoData:
    """Dataclass for storing Pareto front/set data with metadata."""

    pareto_set: np.ndarray
    pareto_front: np.ndarray
    problem_name: str | None = None
    metadata: dict[str, Any] = None

    @property
    def num_solutions(self) -> int:
        """Number of solutions in the Pareto set/front."""
        return len(self.pareto_set)

    def validate_shapes(self) -> bool:
        """Verify that set and front have matching dimensions."""
        return self.pareto_set.shape[0] == self.pareto_front.shape[0]


class ParetoDataManager:
    """
    Comprehensive manager for Pareto data operations including loading, saving,
    and normalization with configurable strategies.
    """

    def __init__(
        self,
        base_path: str | Path = "data/raw",
    ):
        """
        Initialize Pareto data manager.

        Args:
            base_path: Base directory for data operations
            default_normalization_fn: Default normalization function
        """
        self.base_path = Path(base_path)

    def save(
        self,
        pareto_set: np.ndarray,
        pareto_front: np.ndarray,
        problem_name: str = "F1",
        filename: str = "pareto_data.npz",
        metadata: dict | None = None,
        validate: bool = True,
    ) -> None:
        """
        Save Pareto set/front data with metadata in numpy format.

        Args:
            pareto_set: Array of decision vectors (n_solutions, n_dim)
            pareto_front: Array of objective vectors (n_solutions, n_obj)
            problem_name: Identifier for the optimization problem
            filename: Target filename for saving
            metadata: Additional metadata to store
            validate: Verify set/front dimension consistency
        """
        # Create data object for validation
        data = ParetoData(
            pareto_set=pareto_set,
            pareto_front=pareto_front,
            problem_name=problem_name,
            metadata=metadata,
        )

        # Validate shape consistency
        if validate and not data.validate_shapes():
            raise ValueError("Pareto set and front have mismatched dimensions")

        # Ensure directory exists
        save_path = self.base_path / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as compressed numpy archive
        np.savez_compressed(
            save_path,
            pareto_set=pareto_set,
            pareto_front=pareto_front,
            problem_name=problem_name,
            metadata=metadata if metadata else {},
        )

    def load(
        self,
        filename: str = "pareto_data.npz",
        normalize: Callable[[np.ndarray], np.ndarray] | None = None,
        validate: bool = True,
        verbose: bool = False,
    ) -> ParetoData:
        """
        Load Pareto data from numpy archive with optional normalization.

        Args:
            filename: Source filename to load
            normalize: Normalization control:
                None: No normalization (default)
                True: Use default normalization
                Callable: Use this normalization function
            validate: Verify set/front dimension consistency
            verbose: Print loading information

        Returns:
            ParetoData: Structured Pareto dataset
        """
        load_path = self.base_path / filename
        if not load_path.exists():
            raise FileNotFoundError(f"No Pareto data found at {load_path}")

        with np.load(load_path, allow_pickle=True) as data:
            try:
                pareto_set = data["pareto_set"]
                pareto_front = data["pareto_front"]
                problem_name = str(data["problem_name"])
                metadata = dict(data["metadata"].item())
            except KeyError as e:
                raise ValueError("Invalid Pareto data format") from e

        # Handle normalization based on argument type
        if normalize is not None:
            pareto_set = normalize(pareto_set)
            pareto_front = normalize(pareto_front)

        # Create data object
        pareto_data = ParetoData(
            pareto_set=pareto_set,
            pareto_front=pareto_front,
            problem_name=problem_name,
            metadata=metadata,
        )

        # Validate shape consistency
        if validate and not pareto_data.validate_shapes():
            raise ValueError("Loaded data has mismatched set/front dimensions")

        if verbose:
            print(f"Loaded {pareto_data.num_solutions} solutions from {load_path}")
            if normalize is not None:
                print(
                    f"Applied normalization: {normalize.__name__ if callable(normalize) else 'default'}"
                )

        return pareto_data

    def get_bounds(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute min/max bounds for normalization reference"""
        return (np.min(data, axis=0), np.max(data, axis=0))


def normalize_to_hypercube(
    data: np.ndarray, bounds: tuple[np.ndarray, np.ndarray] = None
) -> np.ndarray:
    """
    Normalize data to unit hypercube [0, 1]^n for fair comparison.

    Args:
        data: Data array (n_samples, n_features)
        bounds: Optional (min_values, max_values) for manual normalization

    Returns:
        Normalized data
    """
    if bounds is None:
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
    else:
        min_vals, max_vals = bounds

    # Avoid division by zero for constant columns
    ranges = np.where(max_vals > min_vals, max_vals - min_vals, 1.0)

    # Normalize: (value - min) / range
    return (data - min_vals) / ranges
