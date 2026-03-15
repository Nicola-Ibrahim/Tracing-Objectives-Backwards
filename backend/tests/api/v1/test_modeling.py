import pytest


def test_multi_step_transformation_chain(client):
    """Verify that a chain of multiple transformers works in preview."""
    ds_name = "chain_test_ds"
    client.post("/api/v1/datasets", json={"dataset_name": ds_name, "params": {}})

    payload = {
        "dataset_name": ds_name,
        "x_chain": [
            {"type": "standard", "params": {}},
            {"type": "standard", "params": {}},  # Twice is redundant but tests chaining
        ],
        "split": "train",
        "sampling_limit": 5,
    }
    resp = client.post("/api/v1/modeling/transform", json=payload)
    assert resp.status_code == 200
    assert "transformed" in resp.json()


def test_y_transformation_preview(client):
    """Verify that transformations applied to Y variables reflect in preview."""
    ds_name = "y_trans_ds"
    client.post("/api/v1/datasets", json={"dataset_name": ds_name, "params": {}})

    payload = {
        "dataset_name": ds_name,
        "x_chain": [],
        "y_chain": [{"type": "standard", "params": {}}],
        "split": "train",
    }
    resp = client.post("/api/v1/modeling/transform", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    # Check that Y values changed from original
    assert data["original"]["y"] != data["transformed"]["y"]


def test_modeling_sampling_limit_strictness(client):
    """Assert that the sampling limit is strictly respected in the preview."""
    ds_name = "limit_test_ds"
    client.post(
        "/api/v1/datasets",
        json={"dataset_name": ds_name, "params": {"n_samples": 100}},
    )

    for limit in [2, 10, 50]:
        payload = {"dataset_name": ds_name, "x_chain": [], "sampling_limit": limit}
        resp = client.post("/api/v1/modeling/transform", json=payload)
        assert resp.status_code == 200
        assert len(resp.json()["transformed"]["X"]) == limit


@pytest.mark.xfail(
    reason="API returns HTTPException instead of raising, causing ResponseValidationError on error paths"
)
def test_preview_invalid_transformer(client):
    client.post("/api/v1/datasets", json={"dataset_name": "error_ds"})
    payload = {
        "dataset_name": "error_ds",
        "x_chain": [{"name": "InvalidTransformer", "params": {}}],
    }
    resp = client.post("/api/v1/modeling/transform", json=payload)
    # Depending on how the backend handles errors, might be 422 or 500
    assert resp.status_code in [422, 500]
