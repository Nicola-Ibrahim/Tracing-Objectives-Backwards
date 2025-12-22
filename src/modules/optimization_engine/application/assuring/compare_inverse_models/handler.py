from ....domain.common.interfaces.base_logger import BaseLogger
from ....domain.common.interfaces.base_visualizer import BaseVisualizer
from ....domain.datasets.entities.dataset import Dataset
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ....domain.modeling.interfaces.base_estimator import ProbabilisticEstimator
from ....domain.modeling.interfaces.base_repository import BaseModelArtifactRepository
from ....workflows.inverse_model_comparison import InverseModelComparator
from ...factories.estimator import EstimatorFactory
from .command import CompareInverseModelsCommand


class CompareInverseModelsHandler:
    """
    Compares inverse model candidates against a forward model (simulator).
    """

    def __init__(
        self,
        processed_data_repository: BaseDatasetRepository,
        model_repository: BaseModelArtifactRepository,
        logger: BaseLogger,
        estimator_factory: EstimatorFactory,
        visualizer: BaseVisualizer | None = None,
    ) -> None:
        self._data_repository = processed_data_repository
        self._model_repository = model_repository
        self._logger = logger
        self._estimator_factory = estimator_factory
        self._visualizer = visualizer

    def execute(self, command: CompareInverseModelsCommand) -> dict[str, dict]:
        forward_type = command.forward_estimator_type.value

        self._logger.log_info(
            f"Starting inverse model comparison. Candidates: {[(c.type.value, c.version) for c in command.candidates]}, Forward: {forward_type}"
        )

        # 1. Load Test Data
        dataset: Dataset = self._data_repository.load(name=command.dataset_name)
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
            dataset_name=command.dataset_name,
        ).estimator

        # 3. Validate Each Inverse Model
        comparator = InverseModelComparator()
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
                        dataset_name=command.dataset_name,
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
                        dataset_name=command.dataset_name,
                    ).estimator
                    display_name = f"{inverse_type} (Latest)"

                if not isinstance(inverse_estimator, ProbabilisticEstimator):
                    self._logger.log_warning(
                        f"Inverse estimator {inverse_type} is not probabilistic. Sampling might not work as expected."
                    )

                inverse_estimators[display_name] = inverse_estimator

                # Validate
                self._logger.log_info(f"Validating {display_name}...")
                results = comparator.validate(
                    inverse_estimator=inverse_estimator,
                    forward_estimator=forward_estimator,
                    test_objectives=test_objectives,
                    decision_normalizer=processed_data.decisions_normalizer,
                    objective_normalizer=processed_data.objectives_normalizer,
                    test_decisions=test_decisions,
                    num_samples=command.num_samples,
                    random_state=command.random_state,
                )
                results_map[display_name] = results

            except Exception as e:
                self._logger.log_error(f"Failed to validate {inverse_type}: {e}")

        # 4. Generate Comparison Plots
        if self._visualizer:
            self._logger.log_info("Generating comparison plots...")
            visualization_data = {
                "results_map": results_map,
            }
            try:
                fig = self._visualizer.plot(visualization_data)
                fig.show()
            except Exception as e:
                self._logger.log_warning(f"Could not display plot: {e}")
        else:
            self._logger.log_info("Visualization skipped (no visualizer provided).")

        return results_map
