from pathlib import Path

from sklearn.model_selection import train_test_split

from ....domain.common.interfaces.base_logger import BaseLogger
from ....domain.datasets.entities.generated_dataset import GeneratedDataset
from ....domain.datasets.entities.processed_dataset import ProcessedDataset
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ....domain.datasets.value_objects.pareto import Pareto
from ...factories.algorithm import AlgorithmFactory
from ...factories.normalizer import NormalizerFactory
from ...factories.optimizer import OptimizerFactory
from ...factories.problem import ProblemFactory
from .generate_dataset_command import GenerateDatasetCommand


class GenerateDatasetCommandHandler:
    """
    Command handler for generating biobjective Pareto data.
    This handler now directly orchestrates the core generation logic,
    without delegating to a separate ParetoGenerationService.
    """

    def __init__(
        self,
        problem_factory: ProblemFactory,
        algorithm_factory: AlgorithmFactory,
        optimizer_factory: OptimizerFactory,
        data_model_repository: BaseDatasetRepository,
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

        # Execute the optimization process
        run_data = optimizer.run()

        self._logger.log_info("Optimization run completed.")

        self._logger.log_info(
            f"Found {run_data.pareto_set.shape[0]} Pareto-optimal solutions."
        )
        self._logger.log_info(
            f"Historical Pareto set contains {run_data.historical_solutions.shape[0]} solutions."
        )

        pareto = Pareto(
            set=run_data.pareto_set,
            front=run_data.pareto_front,
        )

        # Build the GeneratedDataset from the optimization results and original configurations
        generated_dataset = GeneratedDataset(
            name="dataset",
            X=run_data.historical_solutions,
            y=run_data.historical_objectives,
            pareto=pareto,
        )

        processed_dataset = self._build_processed_dataset(
            raw_dataset=generated_dataset,
            command=command,
        )

        # Save both raw and processed variants using the repository
        saved_path = self._data_model_repository.save(
            raw=generated_dataset,
            processed=processed_dataset,
        )
        self._logger.log_info(f"Pareto data saved to: {saved_path}")
        return saved_path

    def _build_processed_dataset(
        self,
        *,
        raw_dataset: GeneratedDataset,
        command: GenerateDatasetCommand,
    ) -> ProcessedDataset:
        """
        Split the raw dataset, fit normalizers, and package the processed bundle.
        """

        X_raw = raw_dataset.X
        y_raw = raw_dataset.y

        X_train, X_test, y_train, y_test = train_test_split(
            X_raw,
            y_raw,
            test_size=command.test_size,
            random_state=command.random_state,
        )

        normalizer_cfg = command.normalizer_config.model_dump()
        X_normalizer = self._normalizer_factory.create(normalizer_cfg)
        y_normalizer = self._normalizer_factory.create(normalizer_cfg)

        X_train_norm = X_normalizer.fit_transform(X_train)
        X_test_norm = X_normalizer.transform(X_test)
        y_train_norm = y_normalizer.fit_transform(y_train)
        y_test_norm = y_normalizer.transform(y_test)

        self._logger.log_info(
            "[postprocess] train shapes X%s y%s | test shapes X%s y%s"
            % (
                X_train_norm.shape,
                y_train_norm.shape,
                X_test_norm.shape,
                y_test_norm.shape,
            )
        )

        return ProcessedDataset.create(
            name=raw_dataset.name,
            X_train=X_train_norm,
            y_train=y_train_norm,
            X_test=X_test_norm,
            y_test=y_test_norm,
            X_normalizer=X_normalizer,
            y_normalizer=y_normalizer,
            pareto=raw_dataset.pareto,
            metadata={
                "source": "generated",
                "test_size": command.test_size,
                "random_state": command.random_state,
                "normalizer": normalizer_cfg,
            },
        )
