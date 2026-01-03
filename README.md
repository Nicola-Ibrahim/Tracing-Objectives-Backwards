# Tracing Objectives Backwards

This project implements a data-driven inverse design framework: learn a mapping from objective space **Y** back to decision space **X**, then use it to propose candidate designs that match a user-specified target objective **Y\***. The focus is **inverse decision mapping** in multi-objective settings, where multiple designs can satisfy the same objective pattern and feasibility must be assessed explicitly.

## Why this exists

Forward simulation answers **X -> Y**. In practice, engineers often need the inverse: **given a target outcome Y\***, what **X** could produce it? This inverse problem is typically ill-posed (one-to-many, unstable, or infeasible). The thesis and codebase formalize a practical workflow that:

- Learns a fast inverse mapper **g: Y -> X** from historical Pareto-optimal data.
- Provides feasibility guidance when targets fall outside the observed Pareto region.
- Supports deterministic and probabilistic inverse models to handle one-to-many mappings.
- Validates candidate decisions via forward checks and ranking in objective space.

## What this project delivers

- **Inverse exploration workflow** for multi-objective problems, with offline training and online querying.
- **Model zoo** spanning deterministic regressors and probabilistic/generative models (e.g., MDN, CVAE, INN).
- **Feasibility and decision validation** (e.g., Pareto proximity, distance-based feasibility, calibration gates).
- **Data generation and evaluation** using synthetic benchmarks (COCO) and domain-specific case studies.
- **Visualization tooling** for datasets, diagnostics, and model comparisons.

## Conceptual pipeline (high level)

1. **Generate or collect Pareto-optimal data** (X, Y).
2. **Train inverse mapper**: learn g such that g(Y) ~= X.
3. **Query-time inverse exploration**:
   - User selects target objective **Y\***.
   - System checks feasibility (bounds + Pareto proximity).
   - Inverse model generates candidate **X\***.
   - Forward check ranks candidates by how well f(X\*) matches Y\*.

## System framework

[See System Framework](docs/processes/system-framework.md)

## Where to start

### Table of contents

- [Inverse design pipeline](docs/processes/inverse-design-pipeline.md)
- [Model training and validation](docs/processes/model-training-validation.md)
- [MDN estimator](docs/modeling/mdn.md)
- [Conditional VAE estimator](docs/modeling/conditional-vae.md)
- [INN estimator](docs/modeling/inn.md)
- [VAE modeling notes](docs/modeling/vae.md)
- [NSGA-II optimization notes](docs/modeling/nsga2-optimization.md)

## Project structure (high level)

- [System design overview](docs/specs/design.md)

## Thesis alignment

This repository supports the thesis **"Tracing the Objectives Backwards: Data-Driven Inverse Exploration of Multi-Objective Problems."** The core contribution is a model-agnostic inverse exploration workflow that learns **Y -> X** mappings from forward evaluations and enables interactive, query-time design without re-running optimization for every target.
