← [Back to Overview](README.md)

# 📊 Dataset Module

**Bounded Context**: Data Generation & Preparation  
**Aggregate Root**: `Dataset`

The `dataset` module is responsible for defining mathematical optimization problems (like BiObj or Electric Vehicle), running evolutionary algorithms to solve them, generating the true Pareto front, and preparing datasets (split, normalized) for surrogate modeling.

## 🥞 DDD Architecture

```mermaid
flowchart LR
    subgraph Domain ["Domain Layer (Core Logic)"]
        direction TB
        E[Dataset<br/>ProcessedData] 
        VO[Pareto]
        I[BaseDataSource]
        I2[BaseRepository<br/>BaseResult<br/>BaseVisualizer]
        S[DatasetGenerationService]
    end

    subgraph Application ["Application Layer (Use Cases)"]
        direction TB
        F[Factories<br/>Algorithm, Optimizer, Problem]
        U1[generation]
        U2[visualization]
    end

    subgraph Infrastructure ["Infrastructure Layer (Details)"]
        direction TB
        A[NSGA-II]
        O[Minimizer]
        P[BiObj, EV Problem]
        DS[OptimizationDataSource]
        OA[BaseAlgorithm<br/>BaseOptimizer<br/>BaseProblem]
        R[FileSystemDatasetRepository]
        V[Plotly Panels]
    end

    Infrastructure -->|Implements Interfaces| Domain
    Application -->|Coordinates| Domain
    Infrastructure -->|Used by| Application

    classDef env fill:#fff,stroke:#333,stroke-width:2px,color:#000;
    class Domain,Application,Infrastructure env;
```

## 📦 Component Inventory

| Layer | Type | Component | Description |
|-------|------|-----------|-------------|
| **Domain** | Entity | `Dataset` | Aggregate root storing raw decisions, objectives, and optional processed data. |
| **Domain** | Entity | `ProcessedData` | Split (train/test) and normalized version of a dataset. |
| **Domain** | Value | `Pareto` | Stores the mathematically derived Pareto front and optimal set. |
| **Domain** | Service | `DatasetGenerationService` | Orchestrates the generation of a dataset from a problem. |
| **Domain** | Interface | `BaseDataSource` | Defines the generic data ingestion contract. |
| **Domain** | Interface | `BaseRepository` | Persistence contract for datasets. |
| **App** | Factory | `AlgorithmFactory` | Constructs specific algorithms (like NSGA-II). |
| **App** | Use Case | `generation` | End-to-end script orchestrating the `DatasetGenerationService`. |
| **Infra** | Adapter | `OptimizationDataSource` | Wraps pymoo pipeline into a BaseDataSource. |
| **Infra** | Interface | `BaseProblem`, `BaseAlgorithm`, `BaseOptimizer` | Framework-specific optimization contracts. |
| **Infra** | Repo | `FileSystemDatasetRepository` | Saves datasets to JSON/NPZ on the local file system. |
| **Infra** | Visual | `Plotly Visualizer` | Concrete implementation of dataset visualizations. |

## 🔗 Entity Relationships

```mermaid
classDiagram
    class Dataset {
        +name: str
        +decisions: ndarray
        +objectives: ndarray
        +pareto: Pareto
        +processed: ProcessedData
    }
    
    class ProcessedData {
        +objectives_train: ndarray
        +objectives_test: ndarray
        +decisions_train: ndarray
        +decisions_test: ndarray
        +objectives_normalizer: BaseNormalizer
        +decisions_normalizer: BaseNormalizer
    }
    
    class Pareto {
        +front: ndarray
        +set: ndarray
    }
    
    Dataset "1" *-- "1" Pareto : contains
    Dataset "1" *-- "0..1" ProcessedData : optionally has
```

## 🔄 Data Flow

1. **Definition**: A `BaseProblem` is defined mathematically.
2. **Solving**: A `BaseAlgorithm` (e.g., NSGA-II) runs on the problem to generate raw candidates and derive the raw `Pareto` optimal boundary.
3. **Packaging**: The raw targets are bundled into the `Dataset` aggregate root.
4. **Processing**: Normalization scaling applies to the features via `BaseNormalizer`, and indices are randomly split into train/test pools, encapsulated in a `ProcessedData` entity child.
5. **Storage**: The dataset is serialized securely locally via the `FileSystemDatasetRepository`.

---
Related: [modeling](modeling.md) | [evaluation](evaluation.md)
