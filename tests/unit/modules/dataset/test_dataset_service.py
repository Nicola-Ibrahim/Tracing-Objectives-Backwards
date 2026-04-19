from unittest.mock import MagicMock

import numpy as np
import pytest

from src.modules.dataset.application.dataset_service import DatasetService
from src.modules.dataset.domain.entities.dataset import Dataset
from src.modules.dataset.domain.value_objects.metadata import DatasetMetadata


@pytest.fixture
def mock_repo():
    return MagicMock()


@pytest.fixture
def mock_engine_repo():
    return MagicMock()


@pytest.fixture
def mock_gen_factory():
    return MagicMock()


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def service(mock_repo, mock_engine_repo, mock_gen_factory, mock_logger):
    return DatasetService(mock_repo, mock_engine_repo, mock_gen_factory, mock_logger)


def test_list_datasets_uses_persisted_metadata(service, mock_repo, mock_engine_repo):
    # Mock dataset with metadata
    X = np.random.rand(10, 2)
    y = np.random.rand(10, 1)
    train_indices = np.arange(8)
    test_indices = np.arange(8, 10)
    metadata = DatasetMetadata(
        n_samples=10, n_train=8, n_test=2, split_ratio=0.2, random_state=42
    )
    dataset = Dataset.create(
        name="ds1",
        X=X,
        y=y,
        metadata=metadata,
        train_indices=train_indices,
        test_indices=test_indices,
    )

    mock_repo.list_all.return_value = ["ds1"]
    mock_repo.load.return_value = dataset
    mock_engine_repo.list_engines.return_value = []

    result = service.list_datasets()
    assert result.is_ok
    summary = result.value[0]
    assert summary["name"] == "ds1"
    assert summary["metadata"]["n_samples"] == 10
    assert summary["n_features"] == 2
    assert summary["n_objectives"] == 1


def test_get_dataset_details_uses_persisted_metadata(
    service, mock_repo, mock_engine_repo
):
    X = np.random.rand(10, 2)
    y = np.random.rand(10, 1)
    train_indices = np.arange(7)
    test_indices = np.arange(7, 10)
    metadata = DatasetMetadata(
        n_samples=10, n_train=7, n_test=3, split_ratio=0.3, random_state=42
    )
    dataset = Dataset.create(
        name="ds1",
        X=X,
        y=y,
        metadata=metadata,
        train_indices=train_indices,
        test_indices=test_indices,
    )

    mock_repo.load.return_value = dataset
    mock_engine_repo.list_engines.return_value = []

    result = service.get_dataset_details("ds1", split="all")
    assert result.is_ok
    details = result.value
    assert details["metadata"]["n_samples"] == 10
    assert details["metadata"]["n_train"] == 7
    assert details["metadata"]["n_test"] == 3
