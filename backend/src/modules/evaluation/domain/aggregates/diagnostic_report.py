from datetime import datetime
from typing import Self
from uuid import uuid4

from pydantic import BaseModel, Field

from ..value_objects.decision_assessment import (
    DecisionSpaceDistributionAssessment,
    DecisionSpaceIntervalAssessment,
)
from ..value_objects.engine import Engine
from ..value_objects.objective_assessment import ObjectiveSpaceAssessment


class DiagnosticReport(BaseModel):
    """
    Aggregate Root — full diagnostic evaluation of a stochastic inverse engine.
    Dual-lens: objective space accuracy + decision space calibration.
    """

    # --- Identity ---
    id: str = Field(default_factory=lambda: str(uuid4()))
    run_number: int | None = None

    # --- Engine ---
    engine: Engine

    # --- Run Conditions ---
    dataset_name: str
    num_samples: int
    created_at: datetime = Field(default_factory=datetime.now)

    # --- Assessments ---
    objective_space: ObjectiveSpaceAssessment
    decision_space: (
        DecisionSpaceDistributionAssessment | DecisionSpaceIntervalAssessment
    )

    @classmethod
    def create(
        cls,
        engine: Engine,
        dataset_name: str,
        num_samples: int,
        objective_space: ObjectiveSpaceAssessment,
        decision_space: DecisionSpaceDistributionAssessment
        | DecisionSpaceIntervalAssessment,
    ) -> Self:
        return cls(
            engine=engine,
            dataset_name=dataset_name,
            num_samples=num_samples,
            objective_space=objective_space,
            decision_space=decision_space,
        )

    @classmethod
    def from_data(cls, report: dict) -> Self:
        """Rehydrate from stored data. Pydantic resolves the discriminated union."""
        return cls(**report)
