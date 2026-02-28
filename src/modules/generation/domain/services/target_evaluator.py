import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..entities.generation_context import GenerationContext
from ..interfaces.base_simplex_locator import BaseSimplexLocator


class TargetLocation(BaseModel):
    """
    Value Object representing the geometric placement and physical coherence
    of a target objective within the context's defined space.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    simplex_indices: np.ndarray = Field(
        ..., description="Indices of the vertices forming the containing simplex"
    )
    simplex_decisions: np.ndarray = Field(
        ..., description="Normalized decision-space vertices of the simplex"
    )
    weights: np.ndarray = Field(..., description="Barycentric weights for the target")
    is_inside: bool = Field(
        ..., description="True if target is strictly inside the simplex"
    )
    is_coherent: bool = Field(
        ..., description="True if the local region meets the coherence threshold"
    )
    pathway: str = Field(..., description="'coherent' or 'incoherent'")


class TargetEvaluator:
    """
    Orchestrates the stateless geometric locator and the rich domain model
    to evaluate a target's position and coherence.
    """

    def __init__(self, context: GenerationContext, locator: BaseSimplexLocator):
        self._context = context
        self._locator = locator

    def evaluate(self, target: np.ndarray) -> TargetLocation:
        # 1. Use the stateless geometric service
        vertices_indices, weights, is_inside = self._locator.locate(
            target=target, space_points=self._context.decision_vertices
        )

        # 2. Ask the context to apply its business rules
        is_coherent = is_inside and self._context.evaluate_coherence(vertices_indices)
        pathway = "coherent" if is_coherent else "incoherent"

        return TargetLocation(
            simplex_indices=vertices_indices,
            simplex_decisions=self._context.decision_vertices[vertices_indices],
            weights=weights,
            is_inside=is_inside,
            is_coherent=is_coherent,
            pathway=pathway,
        )
