# Inverse Mapping System Design

## Purpose

This system supports inverse mapping from objective values (Y) back to decision variables (X). It is built to be research-friendly while keeping production-grade structure: modeling, feasibility checks, and evaluation are isolated from I/O, visualization, and orchestration.

## Architectural style

- **Modular monolith**: the codebase is split into clear modules under `src/modules/` while remaining a single deployable project.
- **Domain-Driven Design (DDD) layering**: `domain`, `application`, and `infrastructure` are explicit, plus `cli` and `workflows` for user-facing entry points.
- **Ports and adapters**: domain logic depends on interfaces, while concrete implementations live in infrastructure.

### Layer responsibilities

- **Domain** (`src/modules/optimization_engine/domain`): entities, value objects, domain services, and core rules (e.g., feasibility, validation, modeling abstractions).
- **Application** (`src/modules/optimization_engine/application`): use cases and orchestration (training, generating, assuring, visualizing).
- **Infrastructure** (`src/modules/optimization_engine/infrastructure`): datasets, modeling adapters, visualization, repositories, processing, and shared logging/ACL.
- **CLI** (`src/modules/optimization_engine/cli`): command entry points for generating, training, visualizing, and assurance tasks.
- **Workflows** (`src/modules/optimization_engine/workflows`): end-to-end pipelines that compose multiple use cases.
- **Shared** (`src/modules/shared`): cross-cutting utilities used by multiple modules.

## Core workflow (high level)

1. **Dataset generation**: produce or load Pareto-optimal data (X, Y).
2. **Inverse modeling**: train inverse estimators that map Y to candidate X.
3. **Feasibility and validation**: check bounds, Pareto proximity, and calibration gates.
4. **Evaluation**: forward-evaluate candidates and score objective match quality.
5. **Visualization**: plot datasets, diagnostics, and model comparisons.

## Why this design

- **Clear separation of concerns** keeps domain logic pure and testable.
- **Replaceable adapters** make it easy to swap datasets, models, and plotting tools.
- **Research to production path**: notebooks and experiments remain exploratory, while `src/` keeps a stable, modular codebase.
- **Scalable evolution**: new models, scoring strategies, and visualization modules can be added without refactoring core logic.

## Capabilities and features

- **Inverse design pipeline** with deterministic and probabilistic model families.
- **Feasibility checks** that guard against out-of-distribution targets.
- **Decision validation and scoring** to rank candidate solutions.
- **Dataset generation and benchmarking** using synthetic and domain-specific problems.
- **Diagnostics and visualization** for model performance and Pareto analysis.
- **CLI-driven workflows** for reproducible training, evaluation, and reporting.

## Strength points

- **Modularity**: domain, application, and infrastructure are isolated.
- **Extensibility**: new estimators and validation strategies plug in cleanly.
- **Reproducibility**: workflows and CLI commands enable repeatable runs.
- **Maintainability**: predictable structure reduces coupling and debugging time.

## Technology stack

- **Language/runtime**: Python 3.12
- **Notebooks**: Jupyter (via `ipykernel`, `nbformat`)
- **Modeling/ML**: scikit-learn, PyTorch, pykrige, HDBSCAN, UMAP
- **Optimization**: pymoo, COCO benchmarking tools
- **Data/visualization**: pandas, numpy, matplotlib, seaborn, Plotly
- **CLI and utilities**: click, tqdm, pydantic, wandb
