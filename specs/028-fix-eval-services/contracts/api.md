# API Contracts: Evaluation Services

## Diagnose Response

The `GET /api/v1/evaluation/diagnose` endpoint output.

**Change**: `ecdf` and `pit` metrics no longer use a shared `x` axis array. They now map engine identifiers to individual `{x, y}` object series.

```json
{
  "dataset_name": "string",
  "engines": ["string"],
  "ecdf": {
    "GBPI_v1": {
      "x": [0.1, 0.2, 0.3],
      "y": [0.4, 0.5, 0.6]
    }
  },
  "pit": {
    "GBPI_v1": {
      "x": [0.1, 0.2, 0.3],
      "y": [0.4, 0.5, 0.6]
    }
  },
  "mace": {
    "GBPI_v1": 0.04
  },
  "warnings": []
}
```

## Compare Models Response

The `POST /api/v1/inverse/compare` endpoint (via `CompareInverseModelCandidatesService`).

**Change**: All arrays must be strictly serialized to standard JSON arrays.

```json
{
  "GBPI_v1": {
    "best_index": 0,
    "best_distance": 0.05,
    "best_decision": [0.1, 0.2, 0.3],
    "best_objective": [0.9, 0.8],
    "all_distances": [0.05, 0.1, 0.15],
    "decisions": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    "predicted_objectives": [[0.9, 0.8], [0.8, 0.7]]
  }
}
```
