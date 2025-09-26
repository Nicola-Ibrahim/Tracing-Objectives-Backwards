from typing import Callable, Dict, Sequence, Type

from ...domain.assurance.decision_validation.interfaces import (
    ConformalCalibrator,
    OODCalibrator,
)
from ...domain.assurance.feasibility.interfaces.diversity import (
    BaseDiversityStrategy,
)
from ...domain.assurance.feasibility.interfaces.scoring import (
    BaseFeasibilityScoringStrategy,
)
from ...infrastructure.assurance.decision_validation.calibration import (
    MahalanobisCalibrator,
    SplitConformalL2Calibrator,
)
from ...infrastructure.assurance.decision_validation.forward_models import (
    ForwardEnsembleAdapter,
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
from .estimator import EstimatorFactory


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

    def create(self, method: str, **params) -> BaseDiversityStrategy:
        """Create a diversity strategy based on the specified method and parameters.

        Args:
            method (str): The diversity method to use.
            **params: Additional parameters required for the strategy.
        Returns:
            BaseDiversityStrategy: An instance of the specified diversity strategy.
        Raises:
            ValueError: If the specified method is not recognized.
        """

        strategy_class = self._registry.get(method)
        if not strategy_class:
            raise ValueError(f"Unknown diversity strategy method: {method}")
        return strategy_class(**params)


class OODCalibratorFactory:
    _registry = {
        "mahalanobis": MahalanobisCalibrator,
    }

    def create(self, **params) -> OODCalibrator:
        """Create an OOD calibrator based on the specified method and parameters.

        Args:
            method (str): The calibration method to use.
            **params: Additional parameters required for the calibrator.
        Returns:
            OODCalibrator: An instance of the specified OOD calibrator.
        Raises:
            ValueError: If the specified method is not recognized.
        """
        calibrator_class = self._registry.get(params.pop("method", None))
        if not calibrator_class:
            raise ValueError(f"Unknown OOD calibrator method: {params.get('method')}")
        return calibrator_class(**params)


class ConformalCalibratorFactory:
    _registry = {
        "split_conformal_l2": SplitConformalL2Calibrator,
    }

    def create(self, **params) -> ConformalCalibrator:
        """Create a conformal calibrator based on the specified method and parameters.

        Args:
            method (str): The calibration method to use.
            **params: Additional parameters required for the calibrator.
        Returns:
            ConformalCalibrator: An instance of the specified conformal calibrator.
        Raises:
            ValueError: If the specified method is not recognized.
        """
        calibrator_class = self._registry.get(params.pop("method", None))

        estimator = EstimatorFactory().create(params.pop("estimator", None))

        if not calibrator_class:
            raise ValueError(
                f"Unknown conformal calibrator method: {params.get('method')}"
            )
        return calibrator_class(estimator=estimator, **params)
