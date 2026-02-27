from pathlib import Path

from pydantic import BaseModel, Field

from ...shared.domain.interfaces.base_logger import BaseLogger
from ..domain.entities.dataset import Dataset
from ..domain.interfaces.base_repository import BaseDatasetRepository
from ..domain.value_objects.pareto import Pareto
from ..infrastructure.sources.pymoo.algorithms import NSGA2Config
from ..infrastructure.sources.pymoo.generate import PymooOptimizationGenerator
from ..infrastructure.sources.pymoo.minimizer import MinimizerConfig
from ..infrastructure.sources.pymoo.problems.cocoex import COCOBiObjectiveProblemConfig


class DatasetConfiguration(BaseModel):
    """Unified configuration for dataset generation."""

    problem_id: int = Field(
        ...,
        ge=1,
        le=56,
        description="The problem ID used within the COCO framework.",
    )
    n_var: int = Field(
        2, gt=0, description="Number of decision variables (x dimensions)."
    )

    population_size: int = Field(200, gt=0, description="Size of the population.")

    generations: int = Field(16, ge=0, description="Number of generations.")
    save_history: bool = Field(
        True, description="Whether to save the optimization history."
    )
    verbose: bool = Field(False, description="Whether to print verbose output.")

    dataset_name: str = Field(..., description="Identifier for the dataset.")


class GenerateDatasetService:
    """Service responsible for wiring factories, services, and persistence."""

    def __init__(
        self,
        data_model_repository: BaseDatasetRepository,
        logger: BaseLogger,
    ):
        """
        Initializes the service with repository and logger.

        Args:
            data_model_repository: Component responsible for saving and loading Pareto data.
            logger: Shared logging interface.
        """
        self._data_model_repository = data_model_repository
        self._logger = logger

    def execute(self, config: DatasetConfiguration) -> Path:
        """
        Executes the service by orchestrating dataset generation.

        Args:
            config: The configuration object for dataset generation.

        Returns:
            Path: The file path where the generated dataset is saved.
        """
        self._logger.log_info(
            f"Starting dataset generation for problem {config.problem_id} "
            f"({config.n_var} variables, 2 objectives)"
        )

        # 1. Instantiate the problem
        # Currently defaults to COCO Bi-Objective Problem
        problem_config = COCOBiObjectiveProblemConfig(
            problem_id=config.problem_id, n_var=config.n_var
        )

        # 2. Instantiate the algorithm
        # Currently defaults to NSGA-II
        algo_config = NSGA2Config(population_size=config.population_size)

        # 3. Instantiate the optimizer
        # Currently defaults to Minimizer
        minimizer_config = MinimizerConfig(
            generations=config.generations,
            save_history=config.save_history,
            verbose=config.verbose,
        )

        # 4. Integrate with data source
        data_source = PymooOptimizationGenerator(
            problem_config=problem_config,
            algorithm_config=algo_config,
            minimizer_config=minimizer_config,
        )

        # 5. Generate raw data
        raw_data = data_source.generate()

        # 6. Create Pareto enrichment if available
        pareto = None
        if raw_data.pareto_set is not None and raw_data.pareto_front is not None:
            pareto = Pareto.create(
                set=raw_data.pareto_set,
                front=raw_data.pareto_front,
            )

        # 7. Create the Dataset aggregate (merged from DatasetGenerationService)
        dataset = Dataset.create(
            name=config.dataset_name,
            decisions=raw_data.decisions,
            objectives=raw_data.objectives,
            pareto=pareto,
        )

        if dataset.pareto is not None:
            self._logger.log_info(
                f"Found {dataset.pareto.num_solutions} Pareto-optimal solutions."
            )

        # 8. Save the dataset
        saved_path = self._data_model_repository.save(dataset)
        self._logger.log_info(f"Dataset saved to: {saved_path}")

        return saved_path
