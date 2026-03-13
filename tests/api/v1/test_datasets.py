def test_list_datasets_success(client, mock_list_datasets_service):
    mock_list_datasets_service.execute.return_value = [
        {
            "name": "test_ds",
            "n_samples": 100,
            "n_features": 2,
            "n_objectives": 2,
            "trained_engines_count": 1,
        }
    ]
    response = client.get("/api/v1/datasets")
    assert response.status_code == 200
    assert response.json() == [
        {
            "name": "test_ds",
            "n_samples": 100,
            "n_features": 2,
            "n_objectives": 2,
            "trained_engines_count": 1,
        }
    ]


def test_get_dataset_details_success(client, mock_get_dataset_details_service):
    mock_get_dataset_details_service.execute.return_value = {
        "name": "test_ds",
        "samples": 1,
        "objectives_dim": 2,
        "decisions_dim": 2,
        "n_train": 1,
        "n_test": 0,
        "X": [[1.0, 2.0]],
        "y": [[3.0, 4.0]],
        "is_pareto": [True],
        "bounds": {"obj_0": [0, 10]},
        "trained_engines": [],
    }
    response = client.get("/api/v1/datasets/test_ds")
    assert response.status_code == 200
    assert response.json()["name"] == "test_ds"
    assert "X" in response.json()


def test_get_dataset_details_not_found(client, mock_get_dataset_details_service):
    mock_get_dataset_details_service.execute.side_effect = FileNotFoundError()
    response = client.get("/api/v1/datasets/unknown")
    assert response.status_code == 404


def test_generate_dataset_success(client, mock_generate_dataset_service):
    mock_generate_dataset_service.execute.return_value = "/path/to/ds"
    payload = {
        "function_id": 1,
        "population_size": 100,
        "n_var": 2,
        "generations": 10,
        "split_ratio": 0.2,
        "random_state": 42,
        "dataset_name": "new_ds",
    }
    response = client.post("/api/v1/datasets/generate", json=payload)
    assert response.status_code == 201
    assert response.json() == {
        "status": "success",
        "name": "new_ds",
        "path": "/path/to/ds",
    }


def test_delete_dataset_success(client, mock_delete_dataset_service):
    mock_delete_dataset_service.execute.return_value = {
        "name": "test_ds",
        "engines_removed": 2,
    }
    response = client.delete("/api/v1/datasets/test_ds")
    assert response.status_code == 200
    assert response.json() == {"name": "test_ds", "engines_removed": 2}
