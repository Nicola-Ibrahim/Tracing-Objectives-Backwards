from enum import Enum

from pydantic import BaseModel, Field


class ProblemType(str, Enum):
    biobj = "biobj"


class AlgorithmType(str, Enum):
    nsga2 = "nsga2"


class OptimizerType(str, Enum):
    minimizer = "minimizer"


class ApplicationProblemConfig(BaseModel):
    problem_id: int = Field(
        5,
        ge=1,
        description="The problem ID used within the COCO framework. Must be >= 55.",
        example=55,
    )
    type: ProblemType = Field(
        default=ProblemType.biobj,
        description="The type of optimization problem to solve.",
        example="biobj",
    )


class ApplicationAlgorithmConfig(BaseModel):
    type: AlgorithmType = Field(
        default=AlgorithmType.nsga2,
        description="The optimization algorithm to be used.",
        example="nsga2",
    )
    population_size: int = Field(200, gt=0, description="Size of the population")


class ApplicationOptimizerConfig(BaseModel):
    type: OptimizerType = Field(
        default=OptimizerType.minimizer,
        description="The optimizer runner strategy.",
        example="minimizer",
    )

    generations: int = Field(
        default=16,
        gt=1,
        description="Number of generations for the optimization.",
        example=16,
    )

    save_history: bool = Field(
        default=False,
        description="Whether to save the optimization history.",
        example=False,
    )


class GenerateParetoCommand(BaseModel):
    problem_config: ApplicationProblemConfig = Field(
        ..., description="Configuration of the optimization problem."
    )
    algorithm_config: ApplicationAlgorithmConfig = Field(
        ..., description="Configuration of the optimization algorithm."
    )
    optimizer_config: ApplicationOptimizerConfig = Field(
        ..., description="Configuration of the optimizer execution."
    )
