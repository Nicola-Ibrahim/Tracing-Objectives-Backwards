# Research: Evaluation Service Fixes

## Decision 1: Frontend vs Backend Alignment for ECDF/PIT Charts

**Context**: The backend currently computes ECDF and PIT curves independently for each engine, returning distinct `x` and `y` arrays for each. The frontend `PerformanceChart` component expects a single shared `x` array and multiple `y` arrays.

**Decision**: Modify the frontend `MetricPlotData` interface and `Charts.tsx` component to accept independent `x` and `y` arrays for each engine dataset. 

**Rationale**: Chart.js fully supports independent `{x, y}` point data for Line charts on a linear scale. Forcing the backend to interpolate ECDF/PIT values onto a shared grid introduces artificial smoothing/distortion of the actual empirical distribution. Keeping the exact points preserves mathematical rigor while minimizing backend complexity.

**Alternatives considered**:
- Interpolating backend ECDF/PIT arrays to a shared 100-point grid. Rejected due to potential loss of fidelity, especially for highly skewed distributions or sparse sample sizes.

## Decision 2: Backend Serialization Error Fix

**Context**: The `compare_candidates.py` service gathers `SelectionResult` objects which contain multi-dimensional generated candidates as `np.ndarray` objects. When this dictionary is returned to FastAPI, standard JSON encoders fail on numpy types, causing 500 Internal Server Errors.

**Decision**: Explicitly cast all numpy arrays to Python lists using `.tolist()` and numpy floats to Python `float()` during the construction of `results_map` inside the `compare` service (specifically in `InverseModelsCandidatesComparator`).

**Rationale**: Fixes the 500 error at the domain/application layer boundary before it hits the API framework, ensuring the API contract is strictly JSON-compatible.

**Alternatives considered**: 
- Using a custom JSON encoder at the FastAPI middleware level. Rejected as it clutters the API layer with domain-specific type handling and isn't standard in this codebase.
