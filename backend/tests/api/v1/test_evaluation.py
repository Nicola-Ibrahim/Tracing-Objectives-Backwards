

def test_evaluation_integration(client):
    """
    Integration test for Evaluation module:
    Dataset Prep -> Engine Training -> Diagnosis -> Performance
    """
    ds_name = "eval_test_ds"
    # 1. Prep
    client.post("/api/v1/datasets", json={
        "dataset_name": ds_name,
        "params": {"n_samples": 60}
    })
    client.post("/api/v1/inverse/train", json={
        "dataset_name": ds_name,
        "solver": {"type": "GBPI", "params": {"n_epochs": 1}}
    })
    
    # 2. Diagnose
    diag_payload = {
        "dataset_name": ds_name,
        "candidates": [{"solver_type": "GBPI"}],
        "num_samples": 10
    }
    resp = client.post("/api/v1/evaluation/diagnose", json=diag_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["dataset_name"] == ds_name
    # The keys in 'engines' or 'profiles' might depend on naming
    assert len(data["engines"]) > 0
    
    # 3. Performance
    perf_payload = {
        "dataset_name": ds_name,
        "engine": {"solver_type": "GBPI"}
    }
    resp = client.post("/api/v1/evaluation/performance", json=perf_payload)
    assert resp.status_code == 200
    assert "insights" in resp.json()

def test_diagnose_non_existent_dataset(client):
    payload = {
        "dataset_name": "ghost_ds",
        "candidates": [{"solver_type": "GBPI"}]
    }
    resp = client.post("/api/v1/evaluation/diagnose", json=payload)
    assert resp.status_code == 404
