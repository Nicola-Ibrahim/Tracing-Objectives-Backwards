from ....domain.common.interfaces.base_logger import BaseLogger
from ....domain.datasets.entities.dataset import Dataset
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ....domain.modeling.interfaces.base_estimator import ProbabilisticEstimator
from ....domain.modeling.interfaces.base_repository import BaseModelArtifactRepository
from ....domain.modeling.services.validation import InverseModelValidator
from ...factories.estimator import EstimatorFactory
from .command import ValidateInverseModelCommand


class ValidateInverseModelHandler:
    """
    Validates an inverse model by sampling candidates and evaluating them
    against a forward model (simulator).
    """

    def __init__(
        self,
        processed_data_repository: BaseDatasetRepository,
        model_repository: BaseModelArtifactRepository,
        logger: BaseLogger,
        estimator_factory: EstimatorFactory,
    ) -> None:
        self._data_repository = processed_data_repository
        self._model_repository = model_repository
        self._logger = logger
        self._estimator_factory = estimator_factory

    def execute(self, command: ValidateInverseModelCommand) -> dict[str, float]:
        inverse_type = command.inverse_estimator_type.value
        forward_type = command.forward_estimator_type.value

        self._logger.log_info(
            f"Starting validation. Inverse: {inverse_type}, Forward: {forward_type}"
        )

        # 1. Load Test Data
        dataset: Dataset = self._data_repository.load(name="dataset")
        if not dataset.processed:
            raise ValueError(
                f"Dataset '{dataset.name}' has no processed data available for validation."
            )
        processed_data = dataset.processed

        test_objectives = processed_data.objectives_test  # objective space

        # 2. Load Inverse Model
        if command.inverse_version_id:
            inverse_artifact = self._model_repository.load(
                estimator_type=inverse_type,
                version_id=command.inverse_version_id,
                mapping_direction="inverse",
            )
        else:
            inverse_artifact = self._model_repository.get_latest_version(
                estimator_type=inverse_type,
                mapping_direction="inverse",
            )
        inverse_estimator = inverse_artifact.estimator
        if not isinstance(inverse_estimator, ProbabilisticEstimator):
            self._logger.log_warning(
                "Inverse estimator is not probabilistic. Sampling might not work as expected."
            )

        # 3. Load Forward Model
        if command.forward_version_id:
            forward_artifact = self._model_repository.load(
                estimator_type=forward_type,
                version_id=command.forward_version_id,
                mapping_direction="forward",
            )
        else:
            forward_artifact = self._model_repository.get_latest_version(
                estimator_type=forward_type,
                mapping_direction="forward",
            )
        forward_model = forward_artifact.estimator

        # 4. Delegate to Domain Service
        validator = InverseModelValidator()
        results = validator.validate(
            inverse_estimator=inverse_estimator,
            forward_model=forward_model,
            test_objectives=test_objectives,
            decision_normalizer=processed_data.decisions_normalizer,  # For inverse: X is decisions
            objective_normalizer=processed_data.objectives_normalizer,  # For inverse: y is objectives
            num_samples=command.num_samples,
            random_state=command.random_state,
        )

        self._logger.log_info(f"Validation Results: {results}")
        return results
