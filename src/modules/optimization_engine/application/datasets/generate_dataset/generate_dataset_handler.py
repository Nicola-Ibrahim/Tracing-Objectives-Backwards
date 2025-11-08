from pathlib import Path

from ....domain.common.interfaces.base_logger import BaseLogger
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ....domain.datasets.services import DatasetGenerationService
from ...factories.algorithm import AlgorithmFactory
from ...factories.normalizer import NormalizerFactory
from ...factories.optimizer import OptimizerFactory
from ...factories.problem import ProblemFactory
from .generate_dataset_command import GenerateDatasetCommand


class GenerateDatasetCommandHandler:
    """Command handler responsible for wiring factories, services, and persistence."""

    def __init__(
        self,
        problem_factory: ProblemFactory,
        algorithm_factory: AlgorithmFactory,
        optimizer_factory: OptimizerFactory,
        data_model_repository: BaseDatasetRepository,
        dataset_service: DatasetGenerationService,
        normalizer_factory: NormalizerFactory,
        logger: BaseLogger,
    ):
        """
        Initializes the command handler with necessary factories and an data_model_repository.

        Args:
            problem_factory: Factory to create problem instances.
            algorithm_factory: Factory to create algorithm instances.
            optimizer_factory: Factory to create optimizer runner instances.
            data_model_repository: Component responsible for saving and loading Pareto data.
        """
        self._problem_factory = problem_factory
        self._algorithm_factory = algorithm_factory
        self._optimizer_factory = optimizer_factory
        self._data_model_repository = data_model_repository
        self._dataset_service = dataset_service
        self._normalizer_factory = normalizer_factory
        self._logger = logger

    def execute(self, command: GenerateDatasetCommand) -> Path:
        """
        Executes the command by directly orchestrating the Pareto data generation.

        Args:
            command: The command object containing all necessary configurations.

        Returns:
            Path: The file path where the generated Pareto data is saved.
        """
        problem_config = command.problem_config.model_dump()
        algorithm_config = command.algorithm_config.model_dump()
        optimizer_config = command.optimizer_config.model_dump()

        # Create domain objects using the provided configurations
        problem = self._problem_factory.create(problem_config)
        algorithm = self._algorithm_factory.create(algorithm_config)

        # Create optimizer runner with its dependencies (problem and algorithm)
        optimizer = self._optimizer_factory.create(
            problem=problem,
            algorithm=algorithm,
            config=optimizer_config,
        )

        problem_name = getattr(problem, "name", "unknown")
        algorithm_name = algorithm.__class__.__name__
        optimizer_name = optimizer.__class__.__name__
        self._logger.log_info(
            f"Starting dataset generation for problem '{problem_name}' "
            f"(algorithm={algorithm_name}, optimizer={optimizer_name})."
        )

        normalizer_cfg = command.normalizer_config.model_dump()
        X_normalizer = self._normalizer_factory.create(normalizer_cfg)
        y_normalizer = self._normalizer_factory.create(normalizer_cfg)

        metadata = {
            "source": "generated",
            "normalizer": normalizer_cfg,
        }

        generated_dataset, processed_dataset = self._dataset_service.generate(
            dataset_name="dataset",
            optimizer=optimizer,
            X_normalizer=X_normalizer,
            y_normalizer=y_normalizer,
            test_size=command.test_size,
            random_state=command.random_state,
            metadata=metadata,
        )

        if generated_dataset.pareto is not None:
            self._logger.log_info(
                f"Found {generated_dataset.pareto.set.shape[0]} Pareto-optimal solutions."
            )

        self._logger.log_info(
            f"Historical Pareto set contains {generated_dataset.X.shape[0]} solutions."
        )

        self._logger.log_info(
            "[postprocess] train shapes X%s y%s | test shapes X%s y%s"
            % (
                processed_dataset.X_train.shape,
                processed_dataset.y_train.shape,
                processed_dataset.X_test.shape,
                processed_dataset.y_test.shape,
            )
        )

        # Save both raw and processed variants using the repository
        saved_path = self._data_model_repository.save(
            raw=generated_dataset,
            processed=processed_dataset,
        )
        self._logger.log_info(f"Pareto data saved to: {saved_path}")
        return saved_path
