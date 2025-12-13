from typing import Callable, Dict, Sequence, Type

from ...domain.assurance.decision_validation.interfaces import (
    BaseConformalValidator,
    BaseOODValidator,
)
from ...domain.assurance.feasibility.interfaces.diversity import (
    BaseDiversityStrategy,
)
from ...domain.assurance.feasibility.interfaces.scoring import (
    BaseFeasibilityScoringStrategy,
)
from ...infrastructure.assurance.decision_validation.validators import (
    MahalanobisOODValidator,
    SplitConformalL2Validator,
)
from ...infrastructure.assurance.diversity import (
    ClosestPointsDiversityStrategy,
    KMeansDiversityStrategy,
    MaxMinDistanceDiversityStrategy,
)
from ...infrastructure.assurance.scoring import (
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


class OODValidatorFactory:
    _registry = {
        "mahalanobis": MahalanobisOODValidator,
    }

    def create(self, config) -> BaseOODValidator:
        """Create an OOD validator based on the specified method and parameters.

        Args:
            method (str): The calibration method to use.
            **config: Additional parameters required for the calibrator.
        Returns:
            BaseOODValidator: An instance of the specified OOD validator.
        Raises:
            ValueError: If the specified method is not recognized.
        """
        validator_class = self._registry.get(config.pop("method", None))
        if not validator_class:
            raise ValueError(f"Unknown OOD validator method: {config.get('method')}")
        return validator_class(**config)


class ConformalValidatorFactory:
    _registry = {
        "split_conformal_l2": SplitConformalL2Validator,
    }

    def create(self, config, **_kwargs) -> BaseConformalValidator:
        """Create a conformal validator based on the specified method and parameters.

        Args:
            method (str): The calibration method to use.
            **config: Additional parameters required for the calibrator.
        Returns:
            BaseConformalValidator: An instance of the specified conformal validator.
        Raises:
            ValueError: If the specified method is not recognized.
        """
        validator_class = self._registry.get(config.pop("method", None))

        if not validator_class:
            raise ValueError(
                f"Unknown conformal validator method: {config.get('method')}"
            )
        return validator_class(**config)


# Backward-compatible names
OODCalibratorFactory = OODValidatorFactory
ConformalCalibratorFactory = ConformalValidatorFactory
