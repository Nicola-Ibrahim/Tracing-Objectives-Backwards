from unittest.mock import MagicMock

import numpy as np
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors

from src.modules.inverse.domain.entities.generation_context import GenerationContext


def test_generation_context_initialization_with_knn():
    """Verify that GenerationContext can be initialized with the objective_knn field."""
    mock_mesh = MagicMock(spec=Delaunay)
    mock_surrogate = MagicMock()
    mock_knn = MagicMock(spec=NearestNeighbors)

    space_points = np.array([[0, 0], [1, 1], [0, 1]])
    decision_vertices = np.array([[0.1, 0.1], [0.9, 0.9], [0.1, 0.9]])

    context = GenerationContext(
        dataset_name="test_dataset",
        space_points=space_points,
        decision_vertices=decision_vertices,
        tau=0.1,
        transforms=[],
        surrogate_estimator=mock_surrogate,
        objective_knn=mock_knn,
        mesh=mock_mesh,
    )

    assert context.objective_knn == mock_knn
    assert context.dataset_name == "test_dataset"


def test_locate_with_knn_fallback():
    """Verify that locate() uses KNN fallback for out-of-mesh points."""
    space_points = np.array([[0, 0], [1, 1], [0, 1]])
    mesh = Delaunay(space_points)

    # Setup KNN to return index 1 for a target near [1, 1]
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(space_points)

    context = GenerationContext(
        dataset_name="test_dataset",
        space_points=space_points,
        decision_vertices=np.random.rand(3, 2),
        tau=1.0,
        transforms=[],
        surrogate_estimator=MagicMock(),
        objective_knn=knn,
        mesh=mesh,
    )

    # Target far outside: [2, 2] should be closest to [1, 1] (index 1)
    target = np.array([2, 2])
    indices, weights, is_simplex, is_coherent, anchor_dists = context.locate(target)

    assert indices == [1]
    assert np.allclose(weights, [1.0])
    assert is_simplex is False
    assert is_coherent is False
    assert len(anchor_dists) == 0


def test_locate_inside_mesh_still_uses_delaunay():
    """Verify that locate() still uses Delaunay for points inside the mesh."""
    space_points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    mesh = Delaunay(space_points)
    knn = NearestNeighbors(n_neighbors=1).fit(space_points)

    context = GenerationContext(
        dataset_name="test_dataset",
        space_points=space_points,
        decision_vertices=np.random.rand(4, 2),
        tau=1.0,
        transforms=[],
        surrogate_estimator=MagicMock(),
        objective_knn=knn,
        mesh=mesh,
    )

    # Target inside: [0.5, 0.5]
    target = np.array([0.5, 0.5])
    indices, weights, is_simplex, is_coherent, anchor_dists = context.locate(target)

    assert is_simplex is True
    assert len(indices) == 3  # 三角形の頂点
    assert np.allclose(np.sum(weights), 1.0)
