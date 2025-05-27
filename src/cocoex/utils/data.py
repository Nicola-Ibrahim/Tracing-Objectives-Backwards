from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class ParetoData:
    """Dataclass for storing Pareto front/set data with metadata."""

    pareto_set: np.ndarray
    pareto_front: np.ndarray
    problem_name: Optional[str] = None
    metadata: dict[str, Any] = None

    @property
    def num_solutions(self) -> int:
        """Number of solutions in the Pareto set/front."""
        return len(self.pareto_set)

    def validate_shapes(self) -> bool:
        """Verify that set and front have matching dimensions."""
        return self.pareto_set.shape[0] == self.pareto_front.shape[0]


def save_pareto_data(
    pareto_set: np.ndarray,
    pareto_front: np.ndarray,
    problem_name: str = "F1",
    save_path: str | Path = "data/raw",
    metadata: Optional[dict] = None,
) -> None:
    """
    Save Pareto set/front data with metadata in numpy format.

    Args:
        pareto_set: Array of decision vectors (n_solutions, n_dim)
        pareto_front: Array of objective vectors (n_solutions, n_obj)
        problem_name: Identifier for the optimization problem
        save_path: Directory to save data files
        metadata: Additional metadata to store
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    np.savez(
        save_path / "pareto_data.npz",
        pareto_set=pareto_set,
        pareto_front=pareto_front,
        problem_name=problem_name,
        metadata=metadata if metadata else {},
    )
    print(f"Saved Pareto data to {save_path.absolute()}")


def load_pareto_data(
    data_path: str | Path = "data/raw", verbose: bool = False
) -> ParetoData:
    """
    Load Pareto data from numpy archive with validation.

    Args:
        data_path: Directory containing saved data
        verbose: Print loading information

    Returns:
        ParetoData: Structured Pareto dataset
    """
    load_path = Path(data_path) / "pareto_data.npz"

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

    if verbose:
        print(f"Loaded {pareto_set.shape[0]} solutions from {load_path}")

    return ParetoData(
        pareto_set=pareto_set,
        pareto_front=pareto_front,
        problem_name=problem_name,
        metadata=metadata,
    )


def normalize_to_hypercube(
    data: np.ndarray, bounds: Optional[tuple[np.ndarray, np.ndarray]] = None
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

    # Calculate ranges with stability check (avoid division by zero)
    ranges = np.where(max_vals > min_vals, max_vals - min_vals, 1.0)

    # Normalize: (value - min) / range
    return (data - min_vals) / ranges
