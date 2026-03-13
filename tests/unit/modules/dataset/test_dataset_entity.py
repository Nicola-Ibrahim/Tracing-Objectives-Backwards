import numpy as np
from src.modules.dataset.domain.entities.dataset import Dataset

from src.modules.dataset.domain.value_objects.metadata import DatasetMetadata

def test_dataset_create_populates_fields():
    X = np.random.rand(100, 2)
    y = np.random.rand(100, 1)
    train_indices = np.arange(80)
    test_indices = np.arange(80, 100)
    
    metadata = DatasetMetadata(
        n_samples=100,
        n_train=80,
        n_test=20,
        split_ratio=0.2,
        random_state=42
    )

    dataset = Dataset.create(
        name="test_ds",
        X=X,
        y=y,
        metadata=metadata,
        train_indices=train_indices,
        test_indices=test_indices
    )
    
    assert dataset.metadata.n_samples == 100
    assert dataset.metadata.n_train == 80
    assert dataset.metadata.n_test == 20
    assert len(dataset.train_indices) == 80
    assert len(dataset.test_indices) == 20

def test_dataset_serialization_persists_metadata():
    X = np.random.rand(5, 2)
    y = np.random.rand(5, 1)
    train_indices = np.arange(5)
    test_indices = np.array([], dtype=int)
    
    metadata = DatasetMetadata(
        n_samples=5,
        n_train=5,
        n_test=0,
        split_ratio=0.0,
        random_state=42
    )
    
    dataset = Dataset.create(
        name="test", 
        X=X, 
        y=y, 
        metadata=metadata,
        train_indices=train_indices,
        test_indices=test_indices
    )
    
    data = dataset.model_dump()
    assert data["metadata"]["n_samples"] == 5
    assert data["metadata"]["n_train"] == 5
    assert data["metadata"]["n_test"] == 0
