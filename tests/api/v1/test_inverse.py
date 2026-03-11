def test_train_engine_success(client, mock_train_service):
    mock_train_service.execute.return_value = {
        "dataset_name": "test_ds",
        "solver_type": "GBPI",
        "engine_version": 1,
        "status": "completed",
        "duration_seconds": 1.5,
        "n_train_samples": 80,
        "n_test_samples": 20,
        "split_ratio": 0.8,
        "loss_history": [],
        "transform_summary": [],
    }
    payload = {
        "dataset_name": "test_ds",
        "solver": {"type": "GBPI", "params": {}},
        "transforms": [],
        "split_ratio": 0.8,
        "random_state": 42,
    }
    response = client.post("/api/v1/inverse/train", json=payload)
    assert response.status_code == 201
    assert response.json()["status"] == "completed"


def test_train_engine_unsupported_solver(client):
    payload = {
        "dataset_name": "test_ds",
        "solver": {"type": "CVAE", "params": {}},
        "transforms": [],
        "split_ratio": 0.8,
        "random_state": 42,
    }
    response = client.post("/api/v1/inverse/train", json=payload)
    assert response.status_code == 422
    assert "not yet implemented" in response.json()["detail"]


def test_generate_candidates_success(client, mock_candidate_gen_service):
    mock_candidate_gen_service.execute.return_value = {
        "solver_type": "GBPI",
        "target_objective": [0.5, 0.5],
        "candidate_decisions": [[1.0, 2.0]],
        "candidate_objectives": [[0.51, 0.49]],
        "best_index": 0,
        "best_objective": [0.51, 0.49],
        "best_decision": [1.0, 2.0],
        "y_space_residuals": [0.01],
        "metadata": {},
    }
    payload = {
        "dataset_name": "test_ds",
        "target_objective": [0.5, 0.5],
        "solver_type": "GBPI",
        "n_samples": 1,
    }
    response = client.post("/api/v1/inverse/generate", json=payload)
    assert response.status_code == 200
    assert "candidate_decisions" in response.json()


def test_list_engines_success(client, mock_list_engines_service):
    mock_list_engines_service.execute.return_value = [
        {"solver_type": "GBPI", "version": 1, "created_at": "2024-01-01T00:00:00"}
    ]
    response = client.get("/api/v1/inverse/engines/test_ds")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["solver_type"] == "GBPI"
