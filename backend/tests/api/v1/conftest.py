import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import shutil
import os

from src.api.main import app
from src.api.v1.datasets.dependencies import get_dataset_repository, get_engine_repository
from src.modules.dataset.infrastructure.repositories.dataset_repository import FileSystemDatasetRepository
from src.modules.inverse.infrastructure.repositories.inverse_mapping_engine_repo import FileSystemInverseMappingEngineRepository

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data."""
    tmp_dir = tmp_path_factory.mktemp("data")
    return tmp_dir

@pytest.fixture(autouse=True)
def setup_test_env(test_data_dir, monkeypatch):
    """Override repositories to use the test data directory."""
    # We create specific repositories pointing to the temp dir
    test_ds_repo = FileSystemDatasetRepository(file_path=test_data_dir)
    # FileSystemDatasetRepository joins ROOT_PATH / file_path if file_path is relative.
    # But since test_data_dir is absolute, it stays absolute.
    
    test_engine_repo = FileSystemInverseMappingEngineRepository()
    # Manually override the base path for engine repo
    test_engine_repo._base_storage_path = test_data_dir / "contexts"
    test_engine_repo._base_storage_path.mkdir(parents=True, exist_ok=True)

    # Force the dataset repo to use the absolute temp path directly
    test_ds_repo.base_path = test_data_dir
    test_ds_repo.base_path.mkdir(parents=True, exist_ok=True)

    # Override dependencies in the FastAPI app
    app.dependency_overrides[get_dataset_repository] = lambda: test_ds_repo
    app.dependency_overrides[get_engine_repository] = lambda: test_engine_repo

    yield

    # Clean up after each test if needed, or rely on tmp_path_factory session cleanup
    app.dependency_overrides.clear()

@pytest.fixture
def client():
    """Provides a TestClient for the FastAPI app with dependency overrides."""
    with TestClient(app) as c:
        yield c
