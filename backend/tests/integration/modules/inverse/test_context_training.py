from unittest.mock import MagicMock

import numpy as np

from src.modules.dataset.domain.entities.dataset import Dataset
from src.modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from src.modules.inverse.domain.entities.inverse_mapping_engine import (
    InverseMappingEngine,
)
from src.modules.inverse.domain.value_objects.transform_pipeline import (
    TransformPipeline,
)
from src.modules.inverse.infrastructure.repositories import (
    inverse_mapping_engine_repo,
)
from src.modules.inverse.infrastructure.solvers.gbpi.gbpi_solver import (
    GBPIInverseSolver,
)

FileSystemInverseMappingEngineRepository = (
    inverse_mapping_engine_repo.FileSystemInverseMappingEngineRepository
)


def test_inverse_engine_training_and_persistence_integration(tmp_path):
    """Verify end-to-end training and persistence of InverseMappingEngine."""
    # Setup paths and repositories
    dataset_name = "integration_test_ds"

    # Mock repositories or use tmp_path
    dataset_repo = FileSystemDatasetRepository()
    engine_repo = FileSystemInverseMappingEngineRepository()
    # Override base path for testing
    engine_repo._base_storage_path = tmp_path

    # Create a dummy dataset
    objectives = np.random.rand(10, 2)
    decisions = np.random.rand(10, 3)
    dataset = Dataset.create(
        name=dataset_name, objectives=objectives, decisions=decisions
    )

    dataset_repo.load = MagicMock(return_value=dataset)

    # 1. Setup Solver and Train
    solver = GBPIInverseSolver()
    X_train, y_train = dataset.get_train_data()
    solver.train(X_train, y_train)

    # 2. Create Engine
    engine = InverseMappingEngine.create(
        dataset_name=dataset_name,
        solver=solver,
        transform_pipeline=TransformPipeline(transforms=[]),
    )

    # 3. Persist Engine
    version = engine_repo.save(engine)
    assert version == 1

    # Verify persistence files exist
    # contexts/<dataset>/GBPI/v1-<timestamp>/
    solver_dir = tmp_path / dataset_name / "GBPI"
    assert solver_dir.exists()

    # Find the version directory
    version_dirs = list(solver_dir.glob("v1-*"))
    assert len(version_dirs) == 1
    engine_dir = version_dirs[0]

    assert (engine_dir / "solver.pkl").exists()
    assert (engine_dir / "metadata.toml").exists()
    assert (engine_dir / "transforms").exists()

    # 4. Verify Loading
    loaded_engine = engine_repo.load(dataset_name, "GBPI", version=version)
    assert loaded_engine.dataset_name == dataset_name
    assert loaded_engine.solver.type() == "GBPI"
    assert loaded_engine.solver.tau == solver.tau

    # Test a simple query on the loaded solver
    target = np.array([[0.5, 0.5]])
    res = loaded_engine.solver.generate(target, n_samples=5)
    assert res.candidates_X.shape == (5, 3)
    assert res.metadata["pathway"] in ["coherent", "incoherent", "extrapolation"]
