# Usage Guide (Setup, Training, Visualization)

This guide covers running the pipeline with `uv` and the included Makefile targets.

## Prerequisites

- Python 3.12+
- `uv` installed (`pipx install uv` or `brew install uv`)
- Optional: `make` (for shortcuts)

## Environment setup with `uv`

```bash
uv sync
```

This creates a local environment and installs dependencies from `pyproject.toml` and `uv.lock`.

Run any module via:

```bash
uv run python -m <module>
```

## Makefile shortcuts

List all available commands:

```bash
make help
```

Common defaults:

- `FUNCTION_ID=5`
- `DATASET_NAME=cocoex_f5`
- `INVERSE_ESTIMATOR=mdn`
- `FORWARD_ESTIMATOR=mdn`

Override them per command:

```bash
make data-generate function-id=2
make model-train-inverse estimator=rbf dataset-name=cocoex_f2
```

### Data generation and visualization

```bash
make data-generate
make data-visualize
```

### Inverse model training

```bash
make model-train-inverse
make model-train-inverse-cv
make model-train-inverse-grid
```

### Forward model training

```bash
make model-train-forward
```

### Decision generation and comparison

```bash
make model-generate-decision
make model-compare-inverse
```

### Assurance calibration

```bash
make assurance-calibrate-validation
```

### Visualization

```bash
make model-visualize-inverse
make model-visualize-forward
```

## Direct CLI entry points

If you prefer explicit module calls (useful for notebooks and scripting):

```bash
uv run python -m src.modules.optimization_engine.cli.generating.generate_dataset --function-id 5
uv run python -m src.modules.optimization_engine.cli.training.train_inverse_model_standard --estimator mdn --dataset-name cocoex_f5
uv run python -m src.modules.optimization_engine.cli.visualizing.visualize_model_performance --estimator mdn --mapping-direction inverse --dataset-name cocoex_f5
```

## Notes

- Artifacts and reports are written by the configured repositories in `src/` (see `src/modules/optimization_engine/infrastructure/`).
- For pipeline concepts and diagrams, start with `docs/processes/inverse-design-pipeline.md`.
