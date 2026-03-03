from datetime import datetime

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..interfaces.base_inverse_mapping_solver import AbstractInverseMappingSolver
from ..value_objects.data_split import DataSplit
from ..value_objects.transform_pipeline import TransformPipeline


class InverseMappingEngine(BaseModel):
    """
    The Aggregate Root. Represents a fully prepared environment for generation,
    agnostic of the underlying inverse strategy.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    dataset_name: str = Field(..., description="Identifier of the source dataset")
    solver: AbstractInverseMappingSolver = Field(
        ..., description="The encapsulated inverse strategy engine"
    )
    transform_pipeline: TransformPipeline = Field(
        default_factory=TransformPipeline,
        description="Encapsulates fitted preprocessing transforms",
    )
    data_split: DataSplit = Field(
        default_factory=DataSplit,
        description="Indices and config of the data split",
    )

    created_at: datetime = Field(default_factory=datetime.now)

    @classmethod
    def create(
        cls,
        dataset_name: str,
        solver: AbstractInverseMappingSolver,
        transform_pipeline: TransformPipeline,
        data_split: DataSplit,
    ) -> "InverseMappingEngine":
        return InverseMappingEngine(
            dataset_name=dataset_name,
            solver=solver,
            transform_pipeline=transform_pipeline,
            data_split=data_split,
        )

    def transform_objective(self, target: np.ndarray) -> np.ndarray:
        """Applies internal transforms to an incoming target objective."""
        return self.transform_pipeline.transform_objectives(target)

    def transform_decision(self, decision: np.ndarray) -> np.ndarray:
        """Applies internal transforms to an incoming decision."""
        return self.transform_pipeline.transform_decisions(decision)

    def detransform_decision(self, decision_norm: np.ndarray) -> np.ndarray:
        """Applies inverse transforms to an incoming decision objective."""
        return self.transform_pipeline.detransform_decisions(decision_norm)

    def detransform_objective(self, objective_norm: np.ndarray) -> np.ndarray:
        """Applies inverse transforms to an incoming objective objective."""
        return self.transform_pipeline.detransform_objectives(objective_norm)
