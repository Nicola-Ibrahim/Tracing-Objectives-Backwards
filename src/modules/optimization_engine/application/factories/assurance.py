from typing import Callable, Dict, Sequence, Type

from ...domain.assurance.decision_validation.interfaces import (
    BaseConformalCalibrator,
    BaseOODCalibrator,
)
from ...domain.assurance.feasibility.interfaces.diversity import (
    BaseDiversityStrategy,
)
from ...domain.assurance.feasibility.interfaces.scoring import (
    BaseFeasibilityScoringStrategy,
)
from ...infrastructure.assurance.decision_validation.calibrators import (
    MahalanobisCalibrator,
    SplitConformalL2Calibrator,
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


class OODCalibratorFactory:
    _registry = {
        "mahalanobis": MahalanobisCalibrator,
    }

    def create(self, config) -> BaseOODCalibrator:
        """Create an OOD calibrator based on the specified method and parameters.

        Args:
            method (str): The calibration method to use.
            **config: Additional parameters required for the calibrator.
        Returns:
            BaseOODCalibrator: An instance of the specified OOD calibrator.
        Raises:
            ValueError: If the specified method is not recognized.
        """
        calibrator_class = self._registry.get(config.pop("method", None))
        if not calibrator_class:
            raise ValueError(f"Unknown OOD calibrator method: {config.get('method')}")
        return calibrator_class(**config)


class ConformalCalibratorFactory:
    _registry = {
        "split_conformal_l2": SplitConformalL2Calibrator,
    }

    def create(self, config, **kwargs) -> BaseConformalCalibrator:
        """Create a conformal calibrator based on the specified method and parameters.

        Args:
            method (str): The calibration method to use.
            **config: Additional parameters required for the calibrator.
        Returns:
            BaseConformalCalibrator: An instance of the specified conformal calibrator.
        Raises:
            ValueError: If the specified method is not recognized.
        """
        calibrator_class = self._registry.get(config.pop("method", None))

        estimator = kwargs.get("estimator")

        if not calibrator_class:
            raise ValueError(
                f"Unknown conformal calibrator method: {config.get('method')}"
            )
        return calibrator_class(estimator=estimator, **config)
