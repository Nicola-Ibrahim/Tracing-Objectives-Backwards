from typing import Any, Protocol


class BaseOptimizationAlgorithm(Protocol):
    def __call__(self, config: Any) -> Any:
        """
        This method should be implemented by subclasses to create and configure
        the optimization algorithm instance.

        Args:
            config: Configuration object containing algorithm parameters.
            *args: Additional positional arguments.
            **kwds: Additional keyword arguments.
        """
        raise NotImplementedError(
            "Subclasses must implement the __call__ method to create the algorithm instance."
        )
