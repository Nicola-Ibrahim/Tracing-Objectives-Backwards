# --- from generate_dataset ---
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from ...shared.domain.interfaces.base_logger import BaseLogger
from ..domain.interfaces.base_repository import BaseDatasetRepository
from ..domain.services.dataset_generation_service import DatasetGenerationService
from ..infrastructure.optimization.data_source import OptimizationDataSource
from .factories.algorithm import AlgorithmFactory
from .factories.optimizer import OptimizerFactory
from .factories.problem import ProblemFactory


class ProblemType(str, Enum):
    biobj = "biobj"


class AlgorithmType(str, Enum):
    nsga2 = "nsga2"


class OptimizerType(str, Enum):
    minimizer = "minimizer"


class ApplicationProblemConfig(BaseModel):
    problem_id: int = Field(
        ...,
        ge=1,
        le=56,
        description="The problem ID used within the COCO framework.",
        examples=[5],
    )
    type: ProblemType = Field(
        ...,
        description="The type of optimization problem to solve.",
        examples=["biobj"],
    )


class ApplicationAlgorithmConfig(BaseModel):
    type: AlgorithmType = Field(
        ...,
        description="The optimization algorithm to be used for solving the problem.",
        examples=["nsga2"],
    )
    population_size: int = Field(
        ...,
        gt=0,
        description="Size of the population in each generation.",
        examples=[200],
    )


class ApplicationOptimizerConfig(BaseModel):
    type: OptimizerType = Field(
        ...,
        description="The optimizer runner strategy.",
        examples=["minimizer"],
    )

    generations: int = Field(
        ...,
        gt=1,
        description="Number of generations for the optimization process.",
        examples=[16],
    )

    save_history: bool = Field(
        ...,
        description="Whether to save the optimization history.",
        examples=[True],
    )
    verbose: bool = Field(
        ...,
        description="Whether to print verbose output during optimization.",
        examples=[True],
    )


class GenerateDatasetParams(BaseModel):
    problem_config: ApplicationProblemConfig = Field(
        ...,
        description="Configuration of the optimization problem.",
        examples=[{"problem_id": 5, "type": "biobj"}],
    )
    algorithm_config: ApplicationAlgorithmConfig = Field(
        ...,
        description="Configuration of the optimization algorithm.",
        examples=[{"type": "nsga2", "population_size": 200}],
    )
    optimizer_config: ApplicationOptimizerConfig = Field(
        ...,
        description="Configuration of the optimizer execution.",
        examples=[
            {
                "type": "minimizer",
                "generations": 16,
                "save_history": True,
                "verbose": True,
            }
        ],
    )
    dataset_name: str = Field(
        ...,
        description="Identifier used when persisting dataset artifacts.",
        examples=["dataset"],
    )


class GenerateDatasetService:
    """Service responsible for wiring factories, services, and persistence."""

    def __init__(
        self,
        problem_factory: ProblemFactory,
        algorithm_factory: AlgorithmFactory,
        optimizer_factory: OptimizerFactory,
        data_model_repository: BaseDatasetRepository,
        dataset_service: DatasetGenerationService,
        logger: BaseLogger,
    ):
        """
        Initializes the service with necessary factories and an data_model_repository.

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
        self._logger = logger

    def execute(self, params: GenerateDatasetParams) -> Path:
        """
        Executes the service by directly orchestrating the Pareto data generation.

        Args:
            params: The parameters object containing all necessary configurations.

        Returns:
            Path: The file path where the generated Pareto data is saved.
        """
        problem_config = params.problem_config.model_dump()
        algorithm_config = params.algorithm_config.model_dump()
        optimizer_config = params.optimizer_config.model_dump()

        # Create domain objects using the provided configurations
        problem = self._problem_factory.create(problem_config)
        algorithm = self._algorithm_factory.create(algorithm_config)

        # Create optimizer runner with its dependencies (problem and algorithm)
        optimizer = self._optimizer_factory.create(
            problem=problem,
            algorithm=algorithm,
            config=optimizer_config,
        )

        data_source = OptimizationDataSource(optimizer=optimizer)

        problem_name = getattr(problem, "name", "unknown")
        self._logger.log_info(
            f"Starting dataset generation for problem '{problem_name}' "
        )

        metadata = {
            "source": "generated",
            "problem_id": params.problem_config.problem_id,
        }

        dataset = self._dataset_service.generate(
            dataset_name=params.dataset_name,
            data_source=data_source,
            metadata=metadata,
        )

        if dataset.pareto is not None:
            self._logger.log_info(
                f"Found {dataset.pareto.set.shape[0]} Pareto-optimal solutions."
            )

        self._logger.log_info(
            f"Historical Pareto set contains {dataset.decisions.shape[0]} solutions."
        )

        # Save the dataset aggregate
        saved_path = self._data_model_repository.save(dataset)
        self._logger.log_info(f"Pareto data saved to: {saved_path}")
        return saved_path
