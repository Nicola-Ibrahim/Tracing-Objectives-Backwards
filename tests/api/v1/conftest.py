from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.v1.datasets.dependencies import get_dataset_service
from src.api.v1.evaluation.dependencies import (
    get_diagnose_service,
    get_performance_service,
)
from src.api.v1.inverse.dependencies import get_inverse_service


@pytest.fixture
def mock_dataset_service():
    return MagicMock()


@pytest.fixture
def mock_inverse_service():
    return MagicMock()


@pytest.fixture
def mock_diagnose_service():
    return MagicMock()


@pytest.fixture
def mock_performance_service():
    return MagicMock()


@pytest.fixture
def client(
    mock_dataset_service,
    mock_inverse_service,
    mock_diagnose_service,
    mock_performance_service,
):
    app.dependency_overrides[get_dataset_service] = lambda: mock_dataset_service
    app.dependency_overrides[get_inverse_service] = lambda: mock_inverse_service
    app.dependency_overrides[get_diagnose_service] = lambda: mock_diagnose_service
    app.dependency_overrides[get_performance_service] = lambda: mock_performance_service

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
