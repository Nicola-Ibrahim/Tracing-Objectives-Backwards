from typing import Callable, Dict, Sequence

from ....evaluation.domain.feasibility.interfaces.scoring import (
    BaseFeasibilityScoringStrategy,
)
from ....evaluation.infrastructure.scoring import (
    ConvexHullScoreStrategy,
    KDEScoreStrategy,
    LocalSphereScoreStrategy,
    MinDistanceScoreStrategy,
)


class ScoreStrategyFactory:
    _registry: Dict[str, Callable[[], BaseFeasibilityScoringStrategy]] = {
        "kde": KDEScoreStrategy,
        "min_distance": MinDistanceScoreStrategy,
        "convex_hull": ConvexHullScoreStrategy,
        "local_sphere": LocalSphereScoreStrategy,
    }

    def create(self, method: str) -> BaseFeasibilityScoringStrategy:
        """Create a scoring strategy based on the specified method.

        Args:
            method (str): The scoring method to use.
        Returns:
            BaseFeasibilityScoringStrategy: An instance of the specified scoring strategy.
        Raises:
            ValueError: If the specified method is not recognized.
        """
        strategy_class = self._registry.get(method)
        if not strategy_class:
            raise ValueError(f"Unknown scoring strategy method: {method}")
        return strategy_class()

    def create_bunch(self, methods: Sequence[str]) -> BaseFeasibilityScoringStrategy:
        """Create a scoring strategy based on the specified method.

        Args:
            methods (Sequence[str]): The scoring methods to use.
        Returns:
            BaseFeasibilityScoringStrategy: An instance of the specified scoring strategy.
        Raises:
            ValueError: If the specified method is not recognized.
        """
        strategies = []
        for method in methods:
            strategy_class = self._registry.get(method)
            if not strategy_class:
                raise ValueError(f"Unknown scoring strategy method: {method}")
            strategies.append(strategy_class())
        return ConvexHullScoreStrategy(strategies)
