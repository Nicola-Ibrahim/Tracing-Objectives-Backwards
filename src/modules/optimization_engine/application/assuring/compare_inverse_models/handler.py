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
from .command import CompareInverseModelsCommand, InverseEstimatorCandidate


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
        """
        Coordinates the comparison of multiple inverse model candidates on a single dataset.
        """
        self._logger.log_info(
            f"Starting inverse model comparison on '{command.dataset_name}'. "
            f"Candidates: {[(c.type.value, c.version) for c in command.candidates]}, "
            f"Forward: {command.forward_estimator_type.value}"
        )

        # 1. Load context: dataset and forward model
        dataset: Dataset = self._data_repository.load(name=command.dataset_name)
        if not dataset.processed:
            raise ValueError(
                f"Dataset '{dataset.name}' has no processed data for validation."
            )

        forward_estimator = self._model_repository.get_latest_version(
            estimator_type=command.forward_estimator_type.value,
            mapping_direction="forward",
            dataset_name=command.dataset_name,
        ).estimator

        # 2. Initialize inverse estimators using private helper
        inverse_estimators = self._initialize_estimators(
            candidates=command.candidates, dataset_name=command.dataset_name
        )

        # 3. Run comparison workflow
        comparator = InverseModelComparator()
        results_map = comparator.compare(
            forward_estimator=forward_estimator,
            inverse_estimators=inverse_estimators,
            test_objectives=dataset.processed.objectives_test,
            decision_normalizer=dataset.processed.decisions_normalizer,
            objective_normalizer=dataset.processed.objectives_normalizer,
            test_decisions=dataset.processed.decisions_test,
            num_samples=command.num_samples,
            random_state=command.random_state,
        )

        # 4. Generate comparison plots
        if self._visualizer and results_map:
            self._logger.log_info("Generating comparison plots...")
            self._visualizer.plot({"results_map": results_map})

        return results_map

    def _initialize_estimators(
        self, candidates: list[InverseEstimatorCandidate], dataset_name: str
    ) -> dict[str, BaseEstimator]:
        """
        Initializes inverse estimator instances from the repository.
        """
        inverse_estimators: dict[str, BaseEstimator] = {}

        for candidate in candidates:
            inverse_type = candidate.type.value
            version = candidate.version
            display_name = (
                f"{inverse_type} (v{version})"
                if version
                else f"{inverse_type} (latest)"
            )

            # Resolve from repository
            artifact = self._model_repository.get_version_by_number(
                estimator_type=inverse_type,
                version=version,
                mapping_direction="inverse",
                dataset_name=dataset_name,
            )

            # Sanity check for probabilistic properties
            if not isinstance(artifact.estimator, ProbabilisticEstimator):
                self._logger.log_warning(
                    f"Estimator {display_name} is not probabilistic. Statistical metrics might be unreliable."
                )

            inverse_estimators[display_name] = artifact.estimator

        return inverse_estimators
