# Data Model: Evaluation Service Fixes

## Frontend Adjustments

### `MetricPlotData`
The structure for visualizing metric results (ECDF, PIT).
**Required Change:** Move from shared X-axis to independent data series.

```typescript
export interface MetricSeries {
  x: number[];
  y: number[];
}

export interface MetricPlotData {
  [engineLabel: string]: MetricSeries;
}
```

## Backend Adjustments

### `SelectionResult`
The data structure yielded by `_select` in the `InverseModelsCandidatesComparator`. 
**Required Change:** Serialize fields to Python floats/lists before API return.

```python
@dataclass
class SelectionResult:
    best_index: int
    best_distance: float       # float() wrapper applied
    best_decision: list        # .tolist() applied
    best_objective: list       # .tolist() applied
    all_distances: list        # .tolist() applied
    sorted_candidates: list    # .tolist() applied
    sorted_objectives: list    # .tolist() applied
```
