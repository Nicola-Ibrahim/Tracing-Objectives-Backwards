from typing import Callable, Dict, Sequence, Type

from ....evaluation.domain.decision_validation.interfaces import (
    BaseConformalValidator,
    BaseOODValidator,
)
from ....evaluation.domain.feasibility.interfaces.diversity import (
    BaseDiversityStrategy,
)
from ....evaluation.infrastructure.decision_validation.validators import (
    MahalanobisOODValidator,
    SplitConformalL2Validator,
)
from ....evaluation.infrastructure.diversity import (
    ClosestPointsDiversityStrategy,
    KMeansDiversityStrategy,
    MaxMinDistanceDiversityStrategy,
)


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


