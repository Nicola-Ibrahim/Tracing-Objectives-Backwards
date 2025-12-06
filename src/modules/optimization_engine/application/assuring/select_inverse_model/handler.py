from ....domain.common.interfaces.base_logger import BaseLogger
from ....domain.datasets.entities.dataset import Dataset
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ....domain.modeling.interfaces.base_estimator import ProbabilisticEstimator
from ....domain.modeling.interfaces.base_repository import BaseModelArtifactRepository
from ....domain.modeling.services.validation import InverseModelValidator
from ...factories.estimator import EstimatorFactory
from .command import SelectInverseModelCommand


class SelectInverseModelHandler:
    """
    Selects the best inverse model by validating and comparing multiple candidates
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

    def execute(self, command: SelectInverseModelCommand) -> dict[str, dict]:
        forward_type = command.forward_estimator_type.value

        self._logger.log_info(
            f"Starting model selection. Candidates: {[t.value for t in command.inverse_estimator_types]}, Forward: {forward_type}"
        )

        # 1. Load Test Data
        dataset: Dataset = self._data_repository.load(name="dataset")
        if not dataset.processed:
            raise ValueError(
                f"Dataset '{dataset.name}' has no processed data available for validation."
            )
        processed_data = dataset.processed

        test_objectives = processed_data.objectives_test  # objective space
        test_decisions = processed_data.decisions_test

        # 2. Load Forward Model
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

        # 3. Validate Each Inverse Model
        validator = InverseModelValidator()
        results_map = {}
        inverse_estimators = {}

        for inverse_type_enum in command.inverse_estimator_types:
            inverse_type = inverse_type_enum.value
            try:
                # Load latest version of each inverse model
                inverse_artifact = self._model_repository.get_latest_version(
                    estimator_type=inverse_type,
                    mapping_direction="inverse",
                )
                inverse_estimator = inverse_artifact.estimator

                if not isinstance(inverse_estimator, ProbabilisticEstimator):
                    self._logger.log_warning(
                        f"Inverse estimator {inverse_type} is not probabilistic. Sampling might not work as expected."
                    )

                inverse_estimators[inverse_type] = inverse_estimator

                # Validate
                self._logger.log_info(f"Validating {inverse_type}...")
                results = validator.validate(
                    inverse_estimator=inverse_estimator,
                    forward_model=forward_model,
                    test_objectives=test_objectives,
                    decision_normalizer=processed_data.decisions_normalizer,
                    objective_normalizer=processed_data.objectives_normalizer,
                    num_samples=command.num_samples,
                    random_state=command.random_state,
                )
                results_map[inverse_type] = results

            except Exception as e:
                self._logger.log_error(f"Failed to validate {inverse_type}: {e}")

        # 4. Generate Comparison Plots
        self._logger.log_info("Generating comparison plots...")
        plots = validator.compare_models(
            results_map=results_map,
            test_objectives=test_objectives,
            test_decisions=test_decisions,  # Needed for calibration
            inverse_estimators=inverse_estimators,  # Needed for calibration
        )

        # 5. Display/Save Plots
        fig = plots
        try:
            fig.show()
        except Exception as e:
            self._logger.log_warning(f"Could not display plot: {e}")

        return results_map
