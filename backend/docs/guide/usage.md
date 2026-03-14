# 🛠️ Usage Guide

This guide explains how to interact with the project's systems, from environment setup to running complex inverse design workflows.

---

## 🏗️ Environment Setup

We use `uv` for lightning-fast, reproducible dependency management.

### Installation
1.  Ensure you have `uv` installed ([Instruction](https://github.com/astral-sh/uv)).
2.  Install dependencies and set up the virtual environment:
    ```bash
    uv sync
    ```

> [!TIP]
> Always run project commands via `uv run <command>` or activate the environment with `source .venv/bin/activate`.

---

## 🚁 Running via Makefile

The `Makefile` provides a high-level interface for the most common research tasks. It wraps complex Python module calls into simple, memorable commands.

### Viewing Available Options
To see the full list of shortcuts and their default parameters:
```bash
make help
```

### The Standard Research Loop

| Step | Command | Why run this? |
|------|---------|---------------|
| **1. Data** | `make data-generate` | Creates the ground-truth Pareto front datasets used for training. |
| **2. Train** | `make model-train-inverse` | Trains an inverse estimator to learn the **Y -> X** mapping. |
| **3. Validate**| `make model-visualize-inverse`| Generates diagnostic plots to check if the model actually learned the mapping. |
| **4. Query** | `make model-generate-decision`| The "End Goal": Propose new designs (X) for a specific target (Y). |

---

## 🔧 Overriding Parameters

Most `make` commands support parameter overrides. This is critical for experimenting with different models and functions.

### Example: Different Functions & Estimators
```bash
# Generate data for COCO function 2
make data-generate function-id=2

# Train a Conditional VAE on that data
make model-train-inverse estimator=cvae dataset-name=cocoex_f2

# Run Cross-Validation to get a more robust performance estimate
make model-train-inverse-cv estimator=cvae dataset-name=cocoex_f2
```

---

## ⌨️ Direct CLI Interface (Developer Mode)

For advanced scenarios, you can skip the Makefile and call the modules directly. This gives you access to every available flag (use `--help` on any module).

### Data Generation
```bash
uv run python -m src.modules.optimization_engine.cli.generating.generate_dataset --function-id 5 --n-samples 500
```

### Training
```bash
uv run python -m src.modules.optimization_engine.cli.training.train_inverse_model_standard \
    --estimator mdn \
    --dataset-name cocoex_f5 \
    --epochs 100
```

### Visualization
```bash
uv run python -m src.modules.optimization_engine.cli.visualizing.visualize_model_performance \
    --estimator mdn \
    --mapping-direction inverse \
    --dataset-name cocoex_f5
```

---

## 📈 Understanding Outputs

- **Artifacts**: Models and metadata are saved to `/models/`.
- **Data**: Generated datasets are saved to `/data/`.
- **Visuals**: Plots and reports are generated in `/reports/`.

For a deep dive into *how* these modules work internally, see the **[Inverse Design Pipeline](../processes/inverse-design-pipeline.md)**.
