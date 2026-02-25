from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RawDataPayload:
    """
    Framework-agnostic container for raw data from any source.

    All data sources must provide decisions (X) and objectives (y).
    Pareto data is optional — only sources that compute it (e.g., optimization
    solvers) will populate these fields.
    """

    decisions: np.ndarray
    """Decision variable matrix, shape (n_samples, n_vars)."""

    objectives: np.ndarray
    """Objective value matrix, shape (n_samples, n_objs)."""

    pareto_set: np.ndarray | None = None
    """Pareto-optimal decisions, shape (n_pareto, n_vars). None if not computed."""

    pareto_front: np.ndarray | None = None
    """Pareto-optimal objectives, shape (n_pareto, n_objs). None if not computed."""


class BaseDataSource(ABC):
    """
    Abstract interface defining the contract for any data provider
    (optimization solver, file reader, simulation, API, etc.).

    Implementations decide whether to compute and return Pareto data.
    """

    @abstractmethod
    def generate(self) -> RawDataPayload:
        """
        Produce raw data from the data source.

        Returns:
            RawDataPayload with decisions, objectives, and optionally Pareto data.
        """
        raise NotImplementedError("Subclasses must implement this method.")
