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
    ConformalCalibratorFactory,
    OODCalibratorFactory,
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
        ood_calibrator_factory: OODCalibratorFactory,
        conformal_calibrator_factory: ConformalCalibratorFactory,
        logger: BaseLogger,
    ) -> None:
        """Initialize the command handler.

        Args:
            processed_data_repository (BaseDatasetRepository): Repository for processed datasets.
            calibration_repository (BaseDecisionValidationCalibrationRepository): Repository for calibration artifacts.
            estimator_factory (Callable[[dict[str, object]], BaseEstimator]): Factory for creating forward models.
            forward_adapter_factory (Callable[[Sequence[BaseEstimator]], BaseEstimator]): Factory for creating forward adapters.
            ood_calibrator_factory (Callable[[float, float], BaseOODCalibrator]): Factory for creating OOD calibrators.
            conformal_calibrator_factory (Callable[[float], BaseConformalCalibrator]): Factory for creating conformal calibrators.
            logger (BaseLogger | None, optional): Logger for tracking progress and issues. Defaults to None.
        """
        self._processed_data_repository = processed_data_repository
        self._calibration_repository = calibration_repository
        self._estimator_factory = estimator_factory
        self._ood_calibrator_factory = ood_calibrator_factory
        self._conformal_calibrator_factory = conformal_calibrator_factory
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

        estimator = self._estimator_factory.create(
            command.estimator_params.model_dump()
        )

        ood_calibrator = self._ood_calibrator_factory.create(
            command.ood_calibrator_params.model_dump()
        )

        conformal_calibrator = self._conformal_calibrator_factory.create(
            command.conformal_calibrator_params.model_dump(), estimator=estimator
        )

        ood_calibrator.fit(X=decisions)
        conformal_calibrator.fit(X=decisions, y=objectives)

        calibration = DecisionValidationCalibration(
            ood_calibrator=ood_calibrator,
            conformal_calibrator=conformal_calibrator,
        )
        self._calibration_repository.save(calibration)

        self._logger.log_info(
            "[assurance] Stored calibration "
            f"estimator={calibration.conformal_calibrator.estimator_type} "
            f"radius={calibration.conformal_calibrator.radius:.4f} "
            f"threshold={calibration.ood_calibrator.threshold:.4f}"
        )
