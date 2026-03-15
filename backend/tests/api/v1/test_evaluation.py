import pytest


def test_comparative_diagnostics(client):
    """Verify diagnostics when comparing multiple engine candidates."""
    ds_name = "comp_eval_ds"
    client.post("/api/v1/datasets", json={"dataset_name": ds_name, "params": {}})

    # Train two different engines (v1 and v2)
    client.post(
        "/api/v1/inverse/train",
        json={
            "dataset_name": ds_name,
            "solver": {"type": "GBPI", "params": {"n_epochs": 1}},
        },
    )
    client.post(
        "/api/v1/inverse/train",
        json={
            "dataset_name": ds_name,
            "solver": {"type": "GBPI", "params": {"n_epochs": 2}},
        },
    )

    diag_payload = {
        "dataset_name": ds_name,
        "candidates": [
            {"solver_type": "GBPI", "version": 1},
            {"solver_type": "GBPI", "version": 2},
        ],
        "num_samples": 5,
    }
    resp = client.post("/api/v1/evaluation/diagnose", json=diag_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["engines"]) == 2
    assert "GBPI (v1)" in data["engines"]
    assert "GBPI (v2)" in data["engines"]


def test_diagnose_scale_methods(client):
    """Verify diagnostics with different scaling methods."""
    ds_name = "scale_eval_ds"
    client.post("/api/v1/datasets", json={"dataset_name": ds_name, "params": {}})
    client.post(
        "/api/v1/inverse/train",
        json={
            "dataset_name": ds_name,
            "solver": {"type": "GBPI", "params": {"n_epochs": 1}},
        },
    )

    for method in ["sd", "mad", "iqr"]:
        diag_payload = {
            "dataset_name": ds_name,
            "candidates": [{"solver_type": "GBPI"}],
            "scale_method": method,
        }
        resp = client.post("/api/v1/evaluation/diagnose", json=diag_payload)
        assert resp.status_code == 200, f"Failed for scale_method={method}"
        # Metrics should contain keys for specific engine
        data = resp.json()
        engine_key = data["engines"][0]
        assert engine_key in data["objective_space"]["metrics"]


@pytest.mark.xfail(
    reason="API returns HTTPException instead of raising, causing ResponseValidationError on error paths"
)
def test_diagnose_non_existent_dataset(client):
    payload = {"dataset_name": "ghost_ds", "candidates": [{"solver_type": "GBPI"}]}
    resp = client.post("/api/v1/evaluation/diagnose", json=payload)
    assert resp.status_code == 404
