def test_diagnose_models_success(client, mock_diagnose_service):
    mock_diagnose_service.execute.return_value = {
        "dataset_name": "test_ds",
        "engines": ["GBPI"],
        "ecdf": {},
        "pit": {},
        "mace": {},
        "warnings": [],
    }
    payload = {
        "dataset_name": "test_ds",
        "candidates": [{"solver_type": "GBPI"}],
        "num_samples": 200,
    }
    response = client.post("/api/v1/evaluation/diagnose", json=payload)
    assert response.status_code == 200
    assert response.json()["dataset_name"] == "test_ds"


def test_check_performance_success(client, mock_performance_service):
    mock_performance_service.execute.return_value = {
        "dataset_name": "test_ds",
        "solver_type": "GBPI",
        "version": 1,
        "insights": {},
    }
    payload = {"dataset_name": "test_ds", "engine": {"solver_type": "GBPI"}}
    response = client.post("/api/v1/evaluation/performance", json=payload)
    assert response.status_code == 200
    assert "insights" in response.json()
