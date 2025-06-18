from enum import Enum

from pydantic import BaseModel, Field


class ProblemType(str, Enum):
    biobj = "biobj"
    # Extend here if needed


class AlgorithmType(str, Enum):
    nsga2 = "nsga2"
    # Extend here if needed


class OptimizerType(str, Enum):
    minimizer = "minimizer"
    # Extend here if needed


class ApplicationProblemConfig(BaseModel):
    id: int = Field(
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


class ApplicationOptimizerConfig(BaseModel):
    type: OptimizerType = Field(
        default=OptimizerType.minimizer,
        description="The optimizer runner strategy.",
        example="minimizer",
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
