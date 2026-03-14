

def test_modeling_integration(client):
    """
    Integration test for Modeling module:
    Transformers Discovery -> Preview Transformation
    """
    # 1. Discovery
    resp = client.get("/api/v1/modeling/transformers")
    assert resp.status_code == 200
    transformers = resp.json()["transformers"]
    assert any(t["name"] == "StandardScaler" for t in transformers)
    
    # 2. Preview (requires a dataset)
    ds_name = "model_test_ds"
    client.post("/api/v1/datasets", json={
        "dataset_name": ds_name,
        "params": {"n_samples": 20}
    })
    
    preview_payload = {
        "dataset_name": ds_name,
        "x_chain": [{"name": "StandardScaler", "params": {}}],
        "y_chain": [],
        "split": "train",
        "sampling_limit": 5
    }
    resp = client.post("/api/v1/modeling/transform", json=preview_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "original" in data
    assert "transformed" in data
    assert len(data["transformed"]["X"]) <= 5

def test_preview_invalid_transformer(client):
    client.post("/api/v1/datasets", json={"dataset_name": "error_ds"})
    payload = {
        "dataset_name": "error_ds",
        "x_chain": [{"name": "InvalidTransformer", "params": {}}]
    }
    resp = client.post("/api/v1/modeling/transform", json=payload)
    # Depending on how the backend handles errors, might be 422 or 500
    assert resp.status_code in [422, 500]
