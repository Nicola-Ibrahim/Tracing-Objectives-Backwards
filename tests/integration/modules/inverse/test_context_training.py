from unittest.mock import MagicMock

import numpy as np
import pytest

from src.modules.dataset.domain.entities.dataset import Dataset
from src.modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from src.modules.generation.application.train_context import (
    TrainContextParams,
    TrainContextService,
)
from src.modules.generation.infrastructure.repositories.context_repo import (
    FileSystemContextRepository,
)


class MockLogger:
    def log_info(self, msg):
        pass

    def log_error(self, msg):
        pass


def test_context_training_and_persistence_integration(tmp_path):
    """Verify end-to-end training and persistence of GenerationContext with KNN."""
    # Setup paths and repositories
    dataset_name = "integration_test_ds"

    # Mock repositories or use tmp_path
    dataset_repo = FileSystemDatasetRepository()
    context_repo = FileSystemContextRepository()
    # Override base path for testing
    context_repo._base_storage_path = tmp_path

    # Create a dummy dataset
    objectives = np.random.rand(10, 2)
    decisions = np.random.rand(10, 3)
    # We use a dummy dataset here, but we need to mock the repo load
    dataset = Dataset.create(
        name=dataset_name, objectives=objectives, decisions=decisions
    )

    dataset_repo.load = MagicMock(return_value=dataset)

    service = TrainContextService(
        dataset_repository=dataset_repo,
        context_repository=context_repo,
        logger=MockLogger(),
    )

    params = TrainContextParams(
        dataset_name=dataset_name,
        k_neighbors=3,
        transforms=[],  # Keep it simple for integration test
    )

    # Execute training
    context = service.execute(params)

    # Verify in-memory
    assert context.objective_knn is not None
    assert context.is_trained is True

    # Verify persistence files exist
    context_dir = tmp_path / dataset_name
    assert (context_dir / "objective_knn.pkl").exists()
    assert (context_dir / "mesh.pkl").exists()
    assert (context_dir / "metadata.toml").exists()

    # Verify loading
    loaded_context = context_repo.load(dataset_name)
    assert loaded_context.objective_knn is not None
    # Test a simple query on the loaded KNN
    dist, ind = loaded_context.objective_knn.kneighbors(context.space_points[:1])
    assert dist[0][0] < 1e-10
