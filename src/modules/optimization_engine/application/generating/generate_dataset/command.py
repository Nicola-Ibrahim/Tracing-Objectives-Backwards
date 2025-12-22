from enum import Enum

from pydantic import BaseModel, Field

from ....domain.modeling.enums.normalizer_type import NormalizerTypeEnum
from ...dtos import NormalizerConfig


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
        example=5,
    )
    type: ProblemType = Field(
        ...,
        description="The type of optimization problem to solve.",
        example="biobj",
    )


class ApplicationAlgorithmConfig(BaseModel):
    type: AlgorithmType = Field(
        ...,
        description="The optimization algorithm to be used for solving the problem.",
        example="nsga2",
    )
    population_size: int = Field(
        ..., gt=0, description="Size of the population in each generation.", example=200
    )


class ApplicationOptimizerConfig(BaseModel):
    type: OptimizerType = Field(
        ...,
        description="The optimizer runner strategy.",
        example="minimizer",
    )

    generations: int = Field(
        ...,
        gt=1,
        description="Number of generations for the optimization process.",
        example=16,
    )

    save_history: bool = Field(
        ...,
        description="Whether to save the optimization history.",
        example=True,
    )
    verbose: bool = Field(
        ...,
        description="Whether to print verbose output during optimization.",
        example=True,
    )


class GenerateDatasetCommand(BaseModel):
    problem_config: ApplicationProblemConfig = Field(
        ..., description="Configuration of the optimization problem."
    )
    algorithm_config: ApplicationAlgorithmConfig = Field(
        ..., description="Configuration of the optimization algorithm."
    )
    optimizer_config: ApplicationOptimizerConfig = Field(
        ..., description="Configuration of the optimizer execution."
    )
    dataset_name: str = Field(
        default="dataset",
        description="Identifier used when persisting dataset artifacts.",
    )
    normalizer_config: NormalizerConfig = Field(
        default_factory=lambda: NormalizerConfig(
            type=NormalizerTypeEnum.HYPERCUBE, params={}
        ),
        description="Normalizer applied to train/test splits during post-processing.",
    )
    test_size: float = Field(
        0.2,
        gt=0.0,
        lt=1.0,
        description="Proportion of samples reserved for evaluation (post-processing).",
    )
    random_state: int = Field(
        42, description="Random seed used for the train/test partition."
    )
