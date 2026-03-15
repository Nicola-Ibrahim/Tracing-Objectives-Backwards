import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.v1.datasets.dependencies import (
    get_dataset_repository,
    get_engine_repository,
)
from src.modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from src.modules.inverse.infrastructure.repositories.inverse_mapping_engine_repo import (
    FileSystemInverseMappingEngineRepository,
)


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data."""
    tmp_dir = tmp_path_factory.mktemp("data")
    return tmp_dir


@pytest.fixture(autouse=True)
def setup_test_env(test_data_dir, monkeypatch):
    """Globally patch repository classes to use the test data directory."""
    from src.modules.dataset.infrastructure.repositories.dataset_repository import (
        FileSystemDatasetRepository,
    )
    from src.modules.inverse.infrastructure.repositories.inverse_mapping_engine_repo import (
        FileSystemInverseMappingEngineRepository,
    )

    # Patch FileSystemDatasetRepository
    orig_ds_init = FileSystemDatasetRepository.__init__

    def new_ds_init(self, *args, **kwargs):
        orig_ds_init(self, *args, **kwargs)
        self.base_path = test_data_dir
        self.base_path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(FileSystemDatasetRepository, "__init__", new_ds_init)

    # Patch FileSystemInverseMappingEngineRepository
    orig_inv_init = FileSystemInverseMappingEngineRepository.__init__

    def new_inv_init(self, *args, **kwargs):
        orig_inv_init(self, *args, **kwargs)
        self._base_storage_path = test_data_dir / "contexts"
        self._base_storage_path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        FileSystemInverseMappingEngineRepository, "__init__", new_inv_init
    )

    yield

    # Clean up dependency overrides if any were added manually
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    """Provides a TestClient for the FastAPI app with dependency overrides."""
    with TestClient(app) as c:
        yield c
