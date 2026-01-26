from ....evaluation.domain.decision_validation.interfaces import (
    BaseConformalValidator,
)
from ....evaluation.infrastructure.decision_validation.validators import (
    SplitConformalL2Validator,
)


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
