from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from ..interfaces.base_transform import BaseTransformer
from ..value_objects.estimator import Estimator
from ..value_objects.evaluation_result import EvaluationResult
from ..value_objects.split_step import SplitStep


class TrainedPipeline(BaseModel):
    """
    Represents a self-contained trained ML pipeline artifact.

    This aggregate root bundles the data split configuration, an ordered list
    of preprocessing transform steps, the fitted estimator (with its training log),
    and the final evaluation results.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, json_encoders={datetime: lambda v: v.isoformat()}
    )

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this pipeline instance.",
    )

    dataset_name: str = Field(
        ...,
        description="Dataset identifier associated with this pipeline.",
    )
    mapping_direction: str = Field(
        default="inverse",
        description="Mapping direction used for training (inverse or forward).",
    )

    split: SplitStep = Field(
        ..., description="The data split configuration used during training."
    )

    transforms: list[BaseTransformer] = Field(
        default_factory=list,
        description="Ordered sequence of fitted preprocessing transforms.",
    )

    estimator: Estimator = Field(
        ..., description="The fitted estimator, its config, and training history."
    )

    evaluation: EvaluationResult = Field(
        ..., description="External evaluation metrics computed on test/cv sets."
    )

    run_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata (e.g., execution environment, runtime).",
    )

    trained_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp indicating when this pipeline was trained.",
    )

    version: int | None = Field(
        default=None, description="Automatically assigned sequential version number."
    )

    def get_decisions_transforms(self) -> list[BaseTransformer]:
        from ..enums.transform_target import TransformTarget

        return [
            t
            for t in self.transforms
            if getattr(t, "target", None)
            in (TransformTarget.DECISIONS, TransformTarget.BOTH, "decisions", "both")
        ]

    def get_objectives_transforms(self) -> list[BaseTransformer]:
        from ..enums.transform_target import TransformTarget

        return [
            t
            for t in self.transforms
            if getattr(t, "target", None)
            in (TransformTarget.OBJECTIVES, TransformTarget.BOTH, "objectives", "both")
        ]
