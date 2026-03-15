from unittest.mock import MagicMock

import numpy as np
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors

from src.modules.inverse.infrastructure.solvers.gbpi.gbpi_solver import (
    GBPIInverseSolver,
)


def test_gbpi_solver_initialization():
    """Verify that GBPIInverseSolver correctly initializes its components during train()."""
    solver = GBPIInverseSolver()
    space_points = np.array([[0, 0], [1, 1], [0, 1]])
    # Decisions are what we want to predict (X in training)
    decisions = np.array([[0.1, 0.1], [0.9, 0.9], [0.1, 0.9]])

    # Train the solver
    solver.train(X=decisions, y=space_points)

    assert solver.objective_knn is not None
    assert solver.mesh is not None
    assert solver.tau > 0
    assert solver.X is not None


def test_locate_in_mesh_with_knn_fallback_logic():
    """Verify that _locate_in_mesh and _get_nearest_neighbor work together for fallback."""
    space_points = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    decisions = np.random.rand(3, 2)

    solver = GBPIInverseSolver()
    solver.train(X=decisions, y=space_points)

    # Target far outside: [2, 2] should be closest to [1, 1] (index 1)
    target = np.array([[2.0, 2.0]])

    is_inside, indices, weights = solver._locate_in_mesh(target[0])
    assert is_inside is False

    nn_indices, nn_weights = solver._get_nearest_neighbor(target)
    assert nn_indices == [1]
    assert np.allclose(nn_weights, [1.0])


def test_locate_inside_mesh_logic():
    """Verify that _locate_in_mesh correctly identifies points inside the Delaunay mesh."""
    space_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    decisions = np.random.rand(4, 2)

    solver = GBPIInverseSolver()
    solver.train(X=decisions, y=space_points)

    # Target inside: [0.5, 0.5]
    target = np.array([0.5, 0.5])
    is_inside, indices, weights = solver._locate_in_mesh(target)

    assert is_inside is True
    assert len(indices) == 3  # Triangle vertices (simplices are triangles in 2D)
    assert np.allclose(np.sum(weights), 1.0)
