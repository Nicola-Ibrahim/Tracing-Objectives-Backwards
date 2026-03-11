from unittest.mock import MagicMock
from src.modules.shared.result import Result

def test_diagnose_models_success(client, mock_diagnose_service):
    # Mocking the result of service.execute(params)
    mock_diag = MagicMock()
    mock_diag.metadata.estimator.type = "GBPI"
    mock_diag.metadata.estimator.version = 1
    mock_diag.accuracy.discrepancy_profile.model_dump.return_value = {"x": [0, 1], "y": [0, 1]}
    mock_diag.reliability.pit_profile.model_dump.return_value = {"x": [0, 1], "y": [0, 1]}
    mock_diag.reliability.calibration_error = 0.01

    mock_diagnose_service.execute.return_value = Result.ok([mock_diag])

    payload = {
        "dataset_name": "test_ds",
        "candidates": [{"solver_type": "GBPI"}],
        "num_samples": 200,
    }
    response = client.post("/api/v1/evaluation/diagnose", json=payload)
    assert response.status_code == 200
    assert response.json()["dataset_name"] == "test_ds"
    assert "GBPI (v1)" in response.json()["engines"]
    assert response.json()["ecdf"]["GBPI (v1)"] == {"x": [0.0, 1.0], "y": [0.0, 1.0]}


def test_check_performance_success(client, mock_performance_service):
    mock_performance_service.execute.return_value = Result.ok({
        "dataset_name": "test_ds",
        "solver_type": "GBPI",
        "version": 1,
        "insights": {},
    })
    payload = {"dataset_name": "test_ds", "engine": {"solver_type": "GBPI"}}
    response = client.post("/api/v1/evaluation/performance", json=payload)
    assert response.status_code == 200
    assert "insights" in response.json()
