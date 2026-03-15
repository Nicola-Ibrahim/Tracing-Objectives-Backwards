

def test_inverse_engine_lifecycle_integration(client):
    """
    Integration test for Inverse Mapping engine:
    Dataset Prep -> Training -> Listing -> Candidates Generation -> Deletion
    """
    # 1. Ensure we have a dataset to train on
    ds_name = "inverse_test_ds"
    gen_payload = {
        "dataset_name": ds_name,
        "generator_type": "coco_pymoo",
        "params": {"n_var": 2, "n_obj": 2, "n_samples": 100},
        "split_ratio": 0.2,
        "random_state": 42
    }
    client.post("/api/v1/datasets", json=gen_payload)
    
    # 2. Train an Engine
    train_payload = {
        "dataset_name": ds_name,
        "solver": {
            "type": "GBPI",
            "params": {
                "n_epochs": 2,  # Very few epochs for fast test
                "batch_size": 32,
                "learning_rate": 0.001
            }
        },
        "transforms": [
            {"name": "StandardScaler", "params": {}}
        ]
    }
    resp = client.post("/api/v1/inverse/train", json=train_payload)
    assert resp.status_code == 201
    train_data = resp.json()
    assert train_data["status"] == "completed"
    assert train_data["solver_type"] == "GBPI"
    version = train_data["engine_version"]
    
    # 3. List Engines
    resp = client.get(f"/api/v1/datasets/{ds_name}/engines")
    assert resp.status_code == 200
    engines = resp.json()
    assert any(e["version"] == version for e in engines)
    
    # 4. Generate Candidates
    # Target objective should be within reasonable bounds
    gen_req = {
        "dataset_name": ds_name,
        "solver_type": "GBPI",
        "version": version,
        "target_objective": [0.5, 0.5],
        "n_samples": 5
    }
    resp = client.post("/api/v1/inverse/generate", json=gen_req)
    assert resp.status_code == 200
    candidates = resp.json()
    assert len(candidates["candidate_decisions"]) == 5
    assert len(candidates["candidate_objectives"]) == 5
    assert candidates["best_index"] is not None
    
    # 5. Delete specific engine
    del_payload = {
        "engines": [
            {"dataset_name": ds_name, "solver_type": "GBPI", "version": version}
        ]
    }
    resp = client.request("DELETE", f"/api/v1/datasets/{ds_name}/engines", json=del_payload)
    assert resp.status_code == 200
    assert resp.json()[0]["status"] == "deleted"

def test_train_engine_dataset_not_found(client):
    payload = {
        "dataset_name": "unknown_ds",
        "solver": {"type": "GBPI", "params": {}},
        "transforms": []
    }
    resp = client.post("/api/v1/inverse/train", json=payload)
    assert resp.status_code == 404
    assert resp.json()["detail"]["error_code"] == "NOT_FOUND"
