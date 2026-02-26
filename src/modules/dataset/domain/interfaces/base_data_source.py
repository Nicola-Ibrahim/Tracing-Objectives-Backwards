from abc import ABC, abstractmethod
from typing import Any


class BaseDataSource(ABC):
    """
    Abstract interface defining the contract for any data provider
    (optimization solver, file reader, simulation, API, etc.).

    Implementations decide whether to compute and return Pareto data.
    """

    @abstractmethod
    def generate(self) -> Any:
        """
        Produce raw data from the data source.

        Returns:
            Any: Raw data from the data source.
        """
        raise NotImplementedError("Subclasses must implement this method.")
