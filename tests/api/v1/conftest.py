from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.v1.datasets.dependencies import (
    get_dataset_details_service,
    get_delete_dataset_service,
    get_generation_dataset_service,
    get_list_datasets_service,
)
from src.api.v1.evaluation.dependencies import (
    get_diagnose_service,
    get_performance_service,
)
from src.api.v1.inverse.dependencies import (
    get_generation_service,
    get_list_engines_service,
    get_train_service,
)


@pytest.fixture
def mock_list_datasets_service():
    return MagicMock()


@pytest.fixture
def mock_get_dataset_details_service():
    return MagicMock()


@pytest.fixture
def mock_delete_dataset_service():
    return MagicMock()


@pytest.fixture
def mock_generate_dataset_service():
    return MagicMock()


@pytest.fixture
def mock_list_engines_service():
    return MagicMock()


@pytest.fixture
def mock_train_service():
    return MagicMock()


@pytest.fixture
def mock_candidate_gen_service():
    return MagicMock()


@pytest.fixture
def mock_diagnose_service():
    return MagicMock()


@pytest.fixture
def mock_performance_service():
    return MagicMock()


@pytest.fixture
def client(
    mock_list_datasets_service,
    mock_get_dataset_details_service,
    mock_delete_dataset_service,
    mock_generate_dataset_service,
    mock_list_engines_service,
    mock_train_service,
    mock_candidate_gen_service,
    mock_diagnose_service,
    mock_performance_service,
):
    app.dependency_overrides[get_list_datasets_service] = (
        lambda: mock_list_datasets_service
    )
    app.dependency_overrides[get_dataset_details_service] = (
        lambda: mock_get_dataset_details_service
    )
    app.dependency_overrides[get_delete_dataset_service] = (
        lambda: mock_delete_dataset_service
    )
    app.dependency_overrides[get_generation_dataset_service] = (
        lambda: mock_generate_dataset_service
    )
    app.dependency_overrides[get_list_engines_service] = (
        lambda: mock_list_engines_service
    )
    app.dependency_overrides[get_train_service] = lambda: mock_train_service
    app.dependency_overrides[get_generation_service] = (
        lambda: mock_candidate_gen_service
    )
    app.dependency_overrides[get_diagnose_service] = lambda: mock_diagnose_service
    app.dependency_overrides[get_performance_service] = lambda: mock_performance_service

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
