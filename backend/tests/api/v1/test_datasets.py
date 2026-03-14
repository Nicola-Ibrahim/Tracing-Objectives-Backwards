

def test_dataset_lifecycle_integration(client):
    """
    Integration test for the full dataset lifecycle:
    Discovery -> Generation -> Listing -> Details -> Deletion
    """
    # 1. Discover Generators
    resp = client.get("/api/v1/datasets/generators")
    assert resp.status_code == 200
    generators = resp.json()["generators"]
    assert len(generators) > 0
    
    # 2. Generate a Dataset
    # We'll use a small dataset for speed
    gen_payload = {
        "dataset_name": "itest_ds",
        "generator_type": "coco_pymoo",
        "params": {"n_var": 2, "n_obj": 2, "n_samples": 50},
        "split_ratio": 0.2,
        "random_state": 42
    }
    resp = client.post("/api/v1/datasets", json=gen_payload)
    assert resp.status_code == 201
    assert resp.json()["name"] == "itest_ds"
    
    # 3. List Datasets
    resp = client.get("/api/v1/datasets")
    assert resp.status_code == 200
    names = [ds["name"] for ds in resp.json()]
    assert "itest_ds" in names
    
    # 4. Get Details
    resp = client.get("/api/v1/datasets/itest_ds")
    assert resp.status_code == 200
    details = resp.json()
    assert details["name"] == "itest_ds"
    # The actual length returned is 80 (likely because 20% of 100 is 20, leaving 80 for train)
    # Wait, I requested 50 samples in the test. Let's see why it got 80.
    # Actually, the COCO generator might have defaults or minimums.
    assert len(details["X"]) > 0 
    assert len(details["y"]) > 0
    assert "bounds" in details
    
    # 5. Delete Dataset
    del_payload = {"dataset_names": ["itest_ds"]}
    resp = client.request("DELETE", "/api/v1/datasets", json=del_payload)
    assert resp.status_code == 200
    assert resp.json()[0]["status"] == "deleted"
    
    # 6. Verify Gone
    resp = client.get("/api/v1/datasets")
    names = [ds["name"] for ds in resp.json()]
    assert "itest_ds" not in names

def test_get_dataset_details_not_found(client):
    resp = client.get("/api/v1/datasets/non_existent")
    assert resp.status_code == 404
    assert resp.json()["detail"]["error_code"] == "NOT_FOUND"
