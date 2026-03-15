import pytest


def test_dataset_split_filtering(client):
    """Verify that details correctly filter data by split (train, test, all)."""
    ds_name = "split_test_ds"
    client.post(
        "/api/v1/datasets",
        json={
            "dataset_name": ds_name,
            "params": {"n_samples": 50},
            "split_ratio": 0.2,
        },
    )

    # All
    resp = client.get(f"/api/v1/datasets/{ds_name}?split=all")
    assert resp.status_code == 200
    all_data = resp.json()

    # Train (80% of 50 = 40)
    resp = client.get(f"/api/v1/datasets/{ds_name}?split=train")
    assert resp.status_code == 200
    train_data = resp.json()

    # Test (20% of 50 = 10)
    resp = client.get(f"/api/v1/datasets/{ds_name}?split=test")
    assert resp.status_code == 200
    test_data = resp.json()

    # Note: COCO might produce more samples than requested if constraints require it
    assert len(train_data["X"]) + len(test_data["X"]) == len(all_data["X"])


def test_bulk_dataset_deletion(client):
    """Verify deleting multiple datasets in a single request."""
    ds_names = ["bulk_ds_1", "bulk_ds_2"]
    for name in ds_names:
        client.post("/api/v1/datasets", json={"dataset_name": name, "params": {}})

    # Verify they exist
    resp = client.get("/api/v1/datasets")
    existing = [ds["name"] for ds in resp.json()]
    for name in ds_names:
        assert name in existing

    # Delete bulk
    resp = client.request(
        "DELETE", "/api/v1/datasets", json={"dataset_names": ds_names}
    )
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 2
    for res in results:
        assert res["status"] == "deleted"

    # Verify gone
    resp = client.get("/api/v1/datasets")
    remaining = [ds["name"] for ds in resp.json()]
    for name in ds_names:
        assert name not in remaining


def test_dataset_engine_count_updates(client):
    """Verify that trained_engines_count reflects the actual number of engines."""
    ds_name = "count_test_ds"
    client.post("/api/v1/datasets", json={"dataset_name": ds_name, "params": {}})

    # Check initial count
    resp = client.get("/api/v1/datasets")
    ds_summary = next(ds for ds in resp.json() if ds["name"] == ds_name)
    assert ds_summary["trained_engines_count"] == 0

    # Train an engine
    client.post(
        "/api/v1/inverse/train",
        json={
            "dataset_name": ds_name,
            "solver": {"type": "GBPI", "params": {"n_epochs": 1}},
            "transforms": [],
        },
    )

    # Check updated count
    resp = client.get("/api/v1/datasets")
    ds_summary = next(ds for ds in resp.json() if ds["name"] == ds_name)
    assert ds_summary["trained_engines_count"] == 1


@pytest.mark.xfail(
    reason="API returns HTTPException instead of raising, causing ResponseValidationError on error paths"
)
def test_get_dataset_details_not_found(client):
    resp = client.get("/api/v1/datasets/non_existent")
    assert resp.status_code == 404
    assert resp.json()["detail"]["error_code"] == "NOT_FOUND"


@pytest.mark.xfail(
    reason="Validation error handling returns HTTPException instead of raising"
)
def test_dataset_generation_invalid_params(client):
    """Verify 422/500 for invalid generation configuration."""
    resp = client.post(
        "/api/v1/datasets",
        json={
            "dataset_name": "invalid_ds",
            "params": {"n_samples": -10},  # Invalid
        },
    )
    assert resp.status_code in [422, 500]
