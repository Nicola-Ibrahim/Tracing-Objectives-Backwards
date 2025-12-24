import numpy as np

from ....domain.assurance.decision_validation.entities.decision_validation_calibration import (
    DecisionValidationCalibration,
)
from ....domain.assurance.decision_validation.interfaces import (
    BaseDecisionValidationCalibrationRepository,
)
from ....domain.common.interfaces.base_logger import BaseLogger
from ....domain.datasets.entities.dataset import Dataset
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ...factories.assurance import (
    ConformalValidatorFactory,
    OODValidatorFactory,
)
from ...factories.estimator import EstimatorFactory
from .command import CalibrateDecisionValidationCommand


class CalibrateDecisionValidationCommandHandler:
    def __init__(
        self,
        *,
        processed_data_repository: BaseDatasetRepository,
        calibration_repository: BaseDecisionValidationCalibrationRepository,
        estimator_factory: EstimatorFactory,
        ood_validator_factory: OODValidatorFactory,
        conformal_validator_factory: ConformalValidatorFactory,
        logger: BaseLogger,
    ) -> None:
        """Initialize the command handler.

        Args:
            processed_data_repository (BaseDatasetRepository): Repository for processed datasets.
            calibration_repository (BaseDecisionValidationCalibrationRepository): Repository for calibration artifacts.
            estimator_factory (Callable[[dict[str, object]], BaseEstimator]): Factory for creating forward models.
            forward_adapter_factory (Callable[[Sequence[BaseEstimator]], BaseEstimator]): Factory for creating forward adapters.
            ood_validator_factory (Callable[[float, float], BaseOODValidator]): Factory for creating OOD validators.
            conformal_validator_factory (Callable[[float], BaseConformalValidator]): Factory for creating conformal validators.
            logger (BaseLogger | None, optional): Logger for tracking progress and issues. Defaults to None.
        """
        self._processed_data_repository = processed_data_repository
        self._calibration_repository = calibration_repository
        self._estimator_factory = estimator_factory
        self._ood_validator_factory = ood_validator_factory
        self._conformal_validator_factory = conformal_validator_factory
        self._logger = logger

    def execute(self, command: CalibrateDecisionValidationCommand) -> None:
        dataset: Dataset = self._processed_data_repository.load(
            name=command.dataset_name
        )
        if not dataset.processed:
            raise ValueError(
                f"Dataset '{dataset.name}' has no processed data available for calibration."
            )
        processed_data = dataset.processed

        decisions = processed_data.decisions_train
        objectives = processed_data.objectives_train

        estimator = self._estimator_factory.create(command.estimator_params)

        ood_validator = self._ood_validator_factory.create(
            command.ood_validator_params.model_dump()
        )

        conformal_validator = self._conformal_validator_factory.create(
            command.conformal_validator_params.model_dump(), estimator=estimator
        )

        ood_validator.fit(X=decisions)
        objectives_pred = np.asarray(estimator.predict(decisions), dtype=float)
        if objectives_pred.ndim == 1:
            objectives_pred = objectives_pred.reshape(-1, 1)
        conformal_validator.fit(y_pred=objectives_pred, y_true=objectives)

        calibration = DecisionValidationCalibration(
            ood_validator=ood_validator,
            conformal_validator=conformal_validator,
        )
        self._calibration_repository.save(calibration)

        self._logger.log_info(
            "[assurance] Stored calibration "
            f"radius={calibration.conformal_validator.radius:.4f} "
            f"threshold={calibration.ood_validator.threshold:.4f}"
        )
