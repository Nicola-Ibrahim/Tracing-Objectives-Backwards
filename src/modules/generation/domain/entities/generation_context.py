from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial import Delaunay

from ....modeling.domain.interfaces.base_transform import BaseTransformer


class GenerationContext(BaseModel):
    """
    Rich Aggregate Root representing the environment for generation.
    It encapsulates state and enforces its own domain rules (normalization and coherence).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    dataset_name: str = Field(..., description="Identifier of the source dataset")
    space_points: np.ndarray = Field(
        ..., description="Points forming the Delaunay mesh (N x 2)"
    )
    # mesh_vertices: np.ndarray = Field(
    #     ...,
    #     description="Normalized objective-space points (N x 2), used as Delaunay vertices",
    # )
    decision_vertices: np.ndarray = Field(
        ..., description="Normalized decision-space points (N x D)"
    )
    tau: float = Field(..., gt=0, description="Coherence threshold")
    transforms: list[tuple[BaseTransformer, Any]] = Field(
        default_factory=list,
        description="Ordered sequence of fitted preprocessing transforms with their targets",
    )
    surrogate_estimator: Any = Field(
        ..., description="The Explicitly Evaluated Surrogate Model Step"
    )
    mesh: Delaunay = Field(
        ..., description="The Delaunay triangulation of the space points"
    )
    is_trained: bool = Field(
        default=True,
        description="Flag indicating if the context has been fully trained",
    )
    created_at: datetime = Field(default_factory=datetime.now)

    def normalize_target(self, target: np.ndarray) -> np.ndarray:
        """Applies internal transforms to an incoming target objective."""
        target_norm = target.copy()
        for t in self.get_objectives_transforms():
            target_norm = t.transform(target_norm)
        return target_norm

    def decision_transform(self, decision: np.ndarray) -> np.ndarray:
        """Applies internal transforms to an incoming decision objective."""
        decision_norm = decision.copy()
        for t in self.get_decisions_transforms():
            decision_norm = t.transform(decision_norm)
        return decision_norm

    def decision_detransform(self, decision_norm: np.ndarray) -> np.ndarray:
        """Applies internal transforms to an incoming decision objective."""
        decision = decision_norm.copy()
        for t in reversed(self.get_decisions_transforms()):
            decision = t.inverse_transform(decision)
        return decision

    def objective_transform(self, objective: np.ndarray) -> np.ndarray:
        """Applies internal transforms to an incoming objective objective."""
        objective_norm = objective.copy()
        for t in self.get_objectives_transforms():
            objective_norm = t.transform(objective_norm)
        return objective_norm

    def objective_detransform(self, objective_norm: np.ndarray) -> np.ndarray:
        """Applies internal transforms to an incoming objective objective."""
        objective = objective_norm.copy()
        for t in reversed(self.get_objectives_transforms()):
            objective = t.inverse_transform(objective)
        return objective

    def evaluate_coherence(self, vertices_indices: list[int]) -> bool:
        """Enforces the coherence domain rule based on the tau threshold
        Checks if all pairwise distances between vertices are less than or equal to tau.
        """
        triangle_vertices = self.decision_vertices[vertices_indices]

        if len(triangle_vertices) < 2:
            return True

        # Calculate pairwise distances between vertices
        diffs = triangle_vertices[:, None, :] - triangle_vertices[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        i, j = np.triu_indices(len(triangle_vertices), k=1)
        pairwise_dists = dists[i, j]
        return bool(np.all(pairwise_dists <= self.tau))

    def locate(self, target: np.ndarray) -> tuple[list[int], np.ndarray, bool]:
        """
        Locates the target within the Delaunay mesh and return the triangle.

        Args:
            target: (1, 2) or (2,) array representing the target objective.
            space_points: (N, 2) array of known points forming the mesh.

        Returns:
            anchor_indices: List of original indices (in `space_points`) forming the local geometry.
            weights: Barycentric weights corresponding to the anchors.
            is_inside: True if the target falls inside the convex hull.
        """
        target = np.asarray(target).reshape(1, -1)
        if target.shape[1] != 2:
            raise ValueError("Target must be a 2D coordinate")

        # find the single triangle that the target landed inside
        simplex_idx = self.mesh.find_simplex(target)[0]

        # if target is inside the mesh, use Delaunay triangulation
        # and return the vertices indices and the barycentric weights
        # if simplex index is positive, means a Triangle was found
        if simplex_idx != -1:
            transform = self.mesh.transform[simplex_idx]
            inv_T = transform[:2, :2]
            r = transform[2, :]

            b = inv_T.dot(target[0] - r)
            weights = np.r_[b, 1.0 - b.sum()]  # barycentric weights
            vertices_indices = self.mesh.simplices[
                simplex_idx
            ].tolist()  # indices of the vertices (corners) forming the triangle
            is_simplex_found = True

            # check if the triangle is coherent (not dangerously stretched out)
            is_coherent = self.evaluate_coherence(vertices_indices)
            return vertices_indices, weights, is_simplex_found, is_coherent

        # if target is outside the mesh, find the nearest point in space_points
        # and return the nearest point index and the barycentric weights
        # if simplex index is negative, means no triangle was found
        else:
            distances = np.linalg.norm(self.space_points - target, axis=1)
            closest_vertex_idx = int(np.argmin(distances))
            vertices_indices = [closest_vertex_idx]
            weights = np.array([1.0])
            is_simplex_found = False
            is_coherent = False
            return vertices_indices, weights, is_simplex_found, is_coherent

    def get_decisions_transforms(self) -> list[BaseTransformer]:
        from ....modeling.domain.interfaces.base_transform import TransformTarget

        return [
            t
            for t, target in self.transforms
            if target in (TransformTarget.DECISIONS, "decisions")
        ]

    def get_objectives_transforms(self) -> list[BaseTransformer]:
        from ....modeling.domain.interfaces.base_transform import TransformTarget

        return [
            t
            for t, target in self.transforms
            if target in (TransformTarget.OBJECTIVES, "objectives")
        ]
