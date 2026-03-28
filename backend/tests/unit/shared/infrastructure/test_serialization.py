import json
from datetime import datetime

import numpy as np

from src.modules.shared.infrastructure.serialization import (
    serialize_diagnostics,
)


def test_serialize_numpy_array():
    data = {"array": np.array([1.0, 2.0, 3.0]), "nested": {"val": np.float64(42.0)}}
    serialized = serialize_diagnostics(data)
    decoded = json.loads(serialized)

    assert decoded["array"] == [1.0, 2.0, 3.0]
    assert decoded["nested"]["val"] == 42.0


def test_serialize_datetime():
    dt = datetime(2024, 1, 1, 12, 0, 0)
    data = {"timestamp": dt}
    serialized = serialize_diagnostics(data)
    decoded = json.loads(serialized)

    # ISO format expected
    assert decoded["timestamp"].startswith("2024-01-01T12:00:00")


def test_serialize_mixed_types():
    data = {
        "status": "done",
        "progress": 100,
        "results": [{"score": np.float64(0.95), "id": 1}, {"score": 0.88, "id": 2}],
        "created_at": datetime.now(),
    }
    serialized = serialize_diagnostics(data)
    decoded = json.loads(serialized)

    assert decoded["status"] == "done"
    assert decoded["results"][0]["score"] == 0.95
    assert "created_at" in decoded
