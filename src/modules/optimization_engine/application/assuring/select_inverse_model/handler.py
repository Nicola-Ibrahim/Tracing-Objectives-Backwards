from ....domain.common.interfaces.base_logger import BaseLogger
from ....domain.datasets.entities.dataset import Dataset
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ....domain.modeling.interfaces.base_estimator import ProbabilisticEstimator
from ....domain.modeling.interfaces.base_repository import BaseModelArtifactRepository
from ....workflows.model_selection_workflow import ModelSelector
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
            f"Starting model selection. Candidates: {[(c.type.value, c.version) for c in command.candidates]}, Forward: {forward_type}"
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
        forward_estimator = self._model_repository.get_latest_version(
            estimator_type=forward_type,
            mapping_direction="forward",
        ).estimator

        # 3. Validate Each Inverse Model
        selector = ModelSelector()
        results_map = {}
        inverse_estimators = {}

        for candidate in command.candidates:
            inverse_type = candidate.type.value
            version_int = candidate.version

            try:
                # Load logic
                if version_int is not None:
                    # Find specific version by integer ID
                    all_versions = self._model_repository.get_all_versions(
                        estimator_type=inverse_type,
                        mapping_direction="inverse",
                    )
                    # Filter for matching version
                    target_artifact = next(
                        (a for a in all_versions if a.version == version_int), None
                    )

                    if not target_artifact:
                        raise ValueError(
                            f"Version {version_int} not found for {inverse_type}"
                        )

                    inverse_estimator = target_artifact.estimator
                    display_name = f"{inverse_type} (v{version_int})"
                else:
                    # Load latest
                    inverse_estimator = self._model_repository.get_latest_version(
                        estimator_type=inverse_type,
                        mapping_direction="inverse",
                    ).estimator
                    display_name = f"{inverse_type} (Latest)"

                if not isinstance(inverse_estimator, ProbabilisticEstimator):
                    self._logger.log_warning(
                        f"Inverse estimator {inverse_type} is not probabilistic. Sampling might not work as expected."
                    )

                inverse_estimators[display_name] = inverse_estimator

                # Validate
                self._logger.log_info(f"Validating {display_name}...")
                results = selector.validate(
                    inverse_estimator=inverse_estimator,
                    forward_estimator=forward_estimator,
                    test_objectives=test_objectives,
                    decision_normalizer=processed_data.decisions_normalizer,
                    objective_normalizer=processed_data.objectives_normalizer,
                    num_samples=command.num_samples,
                    random_state=command.random_state,
                )
                results_map[display_name] = results

            except Exception as e:
                self._logger.log_error(f"Failed to validate {inverse_type}: {e}")

        # 4. Generate Comparison Plots
        self._logger.log_info("Generating comparison plots...")
        plots = selector.compare_models(
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
