from typing import Callable, Dict, Sequence, Type

from ...domain.assurance.decision_validation.interfaces import (
    ConformalCalibrator,
    ForwardModel,
    OODCalibrator,
)
from ...domain.assurance.feasibility.interfaces.diversity import (
    BaseDiversityStrategy,
)
from ...domain.assurance.feasibility.interfaces.scoring import (
    BaseFeasibilityScoringStrategy,
)
from ...domain.modeling.interfaces.base_estimator import BaseEstimator
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


def create_default_scoring_strategy() -> BaseFeasibilityScoringStrategy:
    """Return the primary feasibility scoring strategy instance."""

    return KDEScoreStrategy()


def create_scoring_registry() -> (
    Dict[str, Callable[[], BaseFeasibilityScoringStrategy]]
):
    """Return available scoring strategies keyed by configuration name."""

    return {
        "kde": KDEScoreStrategy,
        "min_distance": MinDistanceScoreStrategy,
        "convex_hull": ConvexHullScoreStrategy,
        "local_sphere": LocalSphereScoreStrategy,
    }


def create_default_diversity_registry() -> Dict[str, Type[BaseDiversityStrategy]]:
    """Return the default registry of diversity strategies."""

    return {
        "euclidean": ClosestPointsDiversityStrategy,
        "kmeans": KMeansDiversityStrategy,
        "max_min_distance": MaxMinDistanceDiversityStrategy,
    }


__all__ = [
    "create_default_scoring_strategy",
    "create_scoring_registry",
    "create_default_diversity_registry",
    "create_ood_calibrator",
    "create_conformal_calibrator",
    "create_forward_model",
]


def create_ood_calibrator(*, percentile: float, cov_reg: float) -> OODCalibrator:
    return MahalanobisCalibrator(percentile=percentile, cov_reg=cov_reg)


def create_conformal_calibrator(*, confidence: float) -> ConformalCalibrator:
    return SplitConformalL2Calibrator(confidence)


def create_forward_model(estimators: Sequence[BaseEstimator]) -> ForwardModel:
    return ForwardEnsembleAdapter(estimators)
