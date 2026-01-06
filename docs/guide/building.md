# 🏗️ Building & Environment Guide

This guide details the technical environment, dependency management, and build processes used in this project.

---

## 🐍 Python Version & Runtime

The project is developed for **Python 3.12**. It leverages modern Python features like advanced type hinting and structural pattern matching.

### Recommended Tooling: `uv`
We use [`uv`](https://github.com/astral-sh/uv), an extremely fast Python package and project manager, to manage our environment.

- **Fastest Sync**: `uv sync` installs all dependencies in a fraction of the time compared to `pip`.
- **Reproducible**: The `uv.lock` file ensures every environment is identical.
- **Managed Runtimes**: `uv` can automatically install the correct Python version if it's missing.

---

## 📦 Dependency Layout

Our dependencies are categorized into several groups in `pyproject.toml`:

### Core Research Stack
- `numpy`, `scipy`, `pandas`: Data manipulation and numerical computation.
- `scikit-learn`: Standard ML models and preprocessing utilities.
- `torch`: Deep learning backend for MDN, CVAE, and INN models.

### Optimization & Domain Tools
- `pymoo`: Multi-objective optimization framework for generating ground truth data.
- `pykrige`: Kriging interpolation for spatial/objective mapping.
- `HDBSCAN`, `UMAP`: Clustering and dimensionality reduction for feasibility analysis.

### Engineering & CLI
- `pydantic`: Data validation and settings management.
- `click`: CLI framework for user interaction.
- `tqdm`: Progress bars for long-running training/generation tasks.
- `logging`: Structured logging for traceability.

---

## 📂 Project Directory Structure

```plaintext
/
├── .github/          # CI/CD workflows
├── data/             # Local storage for datasets (ignored by git)
├── docs/             # Comprehensive documentation portal
├── models/           # Local storage for trained model artifacts
├── notebooks/        # Exploratory research and experiments
├── src/              # Source code (Modular Monolith)
│   ├── modules/
│   │   └── optimization_engine/  # Core domain logic
│   └── shared/       # Cross-cutting utilities
├── Makefile          # Developer shortcuts
└── pyproject.toml    # Project configuration and metadata
```

---

## 🛠 Build Workflow

### 1. Synchronization
Whenever dependencies change or after a fresh clone:
```bash
uv sync
```

### 2. Linting & Formatting
We use `ruff` for all-in-one fast linting and formatting. It is configured in `pyproject.toml`.
```bash
# Run linter
uv run ruff check .

# Run auto-formatter
uv run ruff format .
```

### 3. Makefile Integration
The `Makefile` acts as a facade over complex `uv run` commands, making the common workflow easier:
- `make data-generate`: Triggers the dataset generation CLI.
- `make model-train-inverse`: Triggers the training orchestrator.
- `make model-visualize-inverse`: Launches the visualization suite.

---

## 🛑 Common Build Issues

### "Python 3.12 not found"
If `uv` fails to find Python 3.12, you can ask it to manage it for you:
```bash
uv python install 3.12
```

### "Missing dependencies"
If you get `ModuleNotFoundError`, ensure you are running your commands via `uv run` or that your virtual environment is activated.
```bash
uv run python -m ...
# OR
source .venv/bin/activate
python -m ...
```
