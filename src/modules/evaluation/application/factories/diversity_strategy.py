from typing import Dict, Sequence, Type

from ....evaluation.domain.feasibility.interfaces.diversity import (
    BaseDiversityStrategy,
)
from ....evaluation.infrastructure.diversity import (
    ClosestPointsDiversityStrategy,
    KMeansDiversityStrategy,
    MaxMinDistanceDiversityStrategy,
)


class DiversityStrategyFactory:
    _registry: Dict[str, Type[BaseDiversityStrategy]] = {
        "kmeans": KMeansDiversityStrategy,
        "max_min_distance": MaxMinDistanceDiversityStrategy,
        "closest_points": ClosestPointsDiversityStrategy,
    }

    def create(self, method: str, **config) -> BaseDiversityStrategy:
        """Create a diversity strategy based on the specified method and parameters.

        Args:
            method (str): The diversity method to use.
            **config: Additional parameters required for the strategy.
        Returns:
            BaseDiversityStrategy: An instance of the specified diversity strategy.
        Raises:
            ValueError: If the specified method is not recognized.
        """

        strategy_class = self._registry.get(method)
        if not strategy_class:
            raise ValueError(f"Unknown diversity strategy method: {method}")
        return strategy_class(**config)

    def create_bunch(
        self, methods: Sequence[str]
    ) -> BaseDiversityStrategy | list[BaseDiversityStrategy]:
        """Create a diversity strategy based on the specified method and parameters.
        Args:
            methods (Sequence[str]): The diversity methods to use.
        Returns:
            BaseDiversityStrategy: An instance of the specified diversity strategy.
        Raises:
            ValueError: If the specified method is not recognized.
        """

        if methods == ["default"]:
            methods = ["kmeans", "max_min_distance", "closest_points"]

        strategies = []
        for method in methods:
            strategy_class = self._registry.get(method)
            if not strategy_class:
                raise ValueError(f"Unknown diversity strategy method: {method}")
            strategies.append(strategy_class())
        return KMeansDiversityStrategy(n_clusters=len(strategies))
