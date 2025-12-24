from enum import Enum

from pydantic import BaseModel, Field

from ....domain.modeling.enums.normalizer_type import NormalizerTypeEnum
from ....domain.modeling.value_objects.estimator_params import NormalizerConfig


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


class GenerateDatasetCommand(BaseModel):
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
    normalizer_config: NormalizerConfig = Field(
        ...,
        description="Normalizer applied to train/test splits during post-processing.",
        examples=[{"type": NormalizerTypeEnum.HYPERCUBE.value, "params": {}}],
    )
    test_size: float = Field(
        ...,
        gt=0.0,
        lt=1.0,
        description="Proportion of samples reserved for evaluation (post-processing).",
        examples=[0.2],
    )
    random_state: int = Field(
        ..., description="Random seed used for the train/test partition.", examples=[42]
    )
