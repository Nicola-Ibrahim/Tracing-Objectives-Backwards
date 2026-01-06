from ....domain.common.interfaces.base_logger import BaseLogger
from ....domain.common.interfaces.base_visualizer import BaseVisualizer
from ....domain.datasets.entities.dataset import Dataset
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ....domain.modeling.interfaces.base_estimator import (
    BaseEstimator,
    ProbabilisticEstimator,
)
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
            f"Starting inverse model comparison. Candidates: {[(c.type.value, c.version, c.dataset_name) for c in command.candidates]}, Forward: {forward_type}"
        )

        def _load_dataset(dataset_name: str) -> Dataset:
            dataset: Dataset = self._data_repository.load(name=dataset_name)
            if not dataset.processed:
                raise ValueError(
                    f"Dataset '{dataset.name}' has no processed data available for validation."
                )
            return dataset

        def _resolve_forward(dataset_name: str):
            return self._model_repository.get_latest_version(
                estimator_type=forward_type,
                mapping_direction="forward",
                dataset_name=dataset_name,
            ).estimator

        # 1. Group candidates by dataset to avoid redundant loading
        dataset_groups: dict[str, list[any]] = {}
        for candidate in command.candidates:
            ds_name = candidate.dataset_name or "dataset"
            if ds_name not in dataset_groups:
                dataset_groups[ds_name] = []
            dataset_groups[ds_name].append(candidate)

        # 2. Iterate through datasets and compare estimators
        results_map = {}
        comparator = InverseModelComparator()

        for dataset_name, candidates in dataset_groups.items():
            self._logger.log_info(f"Processing comparison for dataset: {dataset_name}")

            try:
                # Load context once per dataset
                dataset = _load_dataset(dataset_name)
                processed_data = dataset.processed
                forward_estimator = _resolve_forward(dataset_name)

                # Initialize all inverse estimators for this dataset
                inverse_estimators: dict[str, BaseEstimator] = {}
                for candidate in candidates:
                    inverse_type = candidate.type.value
                    version_int = candidate.version
                    display_name = f"{inverse_type} (v{version_int})"

                    target_artifact = self._model_repository.get_version_by_number(
                        estimator_type=inverse_type,
                        version=version_int,
                        mapping_direction="inverse",
                        dataset_name=dataset_name,
                    )
                    inverse_estimator = target_artifact.estimator

                    if not isinstance(inverse_estimator, ProbabilisticEstimator):
                        self._logger.log_warning(
                            f"Inverse estimator {inverse_type} is not probabilistic. Sampling might not work as expected."
                        )

                    inverse_estimators[display_name] = inverse_estimator

                # Call the comparator workflow
                dataset_results = comparator.compare(
                    forward_estimator=forward_estimator,
                    inverse_estimators=inverse_estimators,
                    test_objectives=processed_data.objectives_test,
                    decision_normalizer=processed_data.decisions_normalizer,
                    objective_normalizer=processed_data.objectives_normalizer,
                    test_decisions=processed_data.decisions_test,
                    num_samples=command.num_samples,
                    random_state=command.random_state,
                )

                # Merge into global results map, prefixing keys if multiple datasets are involved
                use_prefix = len(dataset_groups) > 1
                for name, res in dataset_results.items():
                    key = f"{name}@{dataset_name}" if use_prefix else name
                    results_map[key] = res

            except Exception as e:
                self._logger.log_error(
                    f"Failed to run comparison for dataset '{dataset_name}': {e}"
                )

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
