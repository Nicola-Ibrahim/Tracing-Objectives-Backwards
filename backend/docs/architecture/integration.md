← [Back to Overview](README.md)

# 🕸 Integration & Dependencies

While Domain-Driven Design emphasizes loose coupling, practical systems inevitably exchange data and contracts across bounded contexts. This page documents all cross-module imports, classifying them into healthy patterns (interface-only) and architectural debt (deep coupling).

## 📊 Integration Matrix

```mermaid
flowchart TD
    dataset[**dataset**]
    modeling[**modeling**]
    evaluation[**evaluation**]
    generation[**generation**]
    shared[**shared**]

    %% Healthy Imports (Green)
    generation -- "BaseModelArtifactRepository<br/>BaseEstimator" --> modeling
    generation -- "BaseDatasetRepository" --> dataset
    
    %% Acceptable Imports (Yellow)
    evaluation -- "Dataset<br/>ProcessedData" --> dataset
    
    %% Debt Imports (Red/Dashed)
    evaluation -.->|"Deep Coupling<br/>(Factories, Enums)"| modeling
    dataset -.->|"Circular Dep<br/>(BaseNormalizer)"| modeling

    %% Styling
    classDef default fill:#f8f9fa,stroke:#b2bec3,stroke-width:2px;
    linkStyle 0,1 stroke:#2ecc71,stroke-width:3px,color:#27ae60;
    linkStyle 2 stroke:#f1c40f,stroke-width:2px,color:#f39c12;
    linkStyle 3,4 stroke:#e74c3c,stroke-width:2px,stroke-dasharray: 5 5,color:#c0392b;
```

## 📦 Cross-Module Import Inventory

Based on an exhaustive scan of the codebase (`grep -r "from \.\.\.\." src/modules/`):

| From Module | To Module | Qualities | Import Example | Count |
|-------------|-----------|-----------|----------------|-------|
| `generation` | `modeling` | 🟢 Interface-only | `BaseModelArtifactRepository`, `BaseEstimator` | 3 |
| `generation` | `dataset` | 🟢 Interface-only | `BaseDatasetRepository` | 1 |
| `generation` | `shared` | 🟢 Utilities | `TomlFileHandler`, `ROOT_PATH` | 2 |
| `evaluation` | `dataset` | 🟡 Entities | `Dataset`, `ProcessedData` | 8 |
| `evaluation` | `shared` | 🟢 Utilities | `JsonFileHandler`, `BaseLogger` | 6 |
| `evaluation` | `modeling` | 🔴 Deep Coupling | `EstimatorFactory`, `EstimatorTypeEnum` | 18+ |
| `dataset` | `modeling` | 🔴 Circular | `BaseNormalizer` | 1 |

> Note: All modules implicitly depend on `shared` for file handling and configuration.

## 🚨 Architectural Debt Log

This log tracks violations of strict DDD boundaries. They represent technical debt that should ideally be paid down in future refactors.

### 1. The `BaseNormalizer` Circular Dependency
- **Symptom**: `dataset.domain.entities.processed_data` imports `BaseNormalizer` from `modeling`. However, `modeling` intrinsically depends on the datasets produced by the `dataset` module.
- **Affected Files**: `src/modules/dataset/domain/entities/processed_data.py`
- **Impact**: Two top-level modules mutually depend on each other, complicating testing, independent deployment (if this were ever split into microservices), and cognitive load.
- **Suggested Fix**: Move `BaseNormalizer` out of `modeling` and into the `shared` module, as normalization is a cross-cutting mathematical operation equally relevant to data prep and model input.

### 2. Misplaced Model Training Orchestration
- **Symptom**: The use case `train_grid_search.py` is located inside the `evaluation` module but is actively used to train models and persist `ModelArtifact`s.
- **Affected Files**: `src/modules/evaluation/application/use_cases/train_grid_search.py`
- **Impact**: The `evaluation` module takes on responsibilities that definitively belong to the `modeling` bounded context.
- **Suggested Fix**: Move this file to `src/modules/modeling/application/use_cases/`.

### 3. Deep Coupling: `evaluation` → `modeling`
- **Symptom**: The `evaluation` module imports highly specific concrete implementations and factories from `modeling` (like `EstimatorTypeEnum`, `EstimatorFactory`, `MetricFactory`).
- **Affected Files**: 
  - `src/modules/evaluation/application/use_cases/compare_candidates.py`
  - `src/modules/evaluation/application/use_cases/diagnose_models.py`
- **Impact**: A small rewrite in how `modeling` instantiates models cascades errors physically throughout the `evaluation` module, breaking encapsulation.
- **Suggested Fix**: Invert the dependency. Have `evaluation` define the interfaces it needs via an Anti-Corruption Layer, or restrict `evaluation`'s imports strictly to `BaseModelArtifactRepository` and `ModelArtifact`.

---
Related: [dataset](dataset.md) | [modeling](modeling.md) | [evaluation](evaluation.md)
