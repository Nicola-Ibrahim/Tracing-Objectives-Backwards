# 🏗️ Inverse Mapping System Design

This document outlines the architectural principles, design decisions, and system capabilities of the inverse mapping framework.

---

## 🎯 Purpose & Vision

The system is designed to provide a robust bridge between **exploratory research** and **stable software engineering**. It supports the inverse mapping from objective values (Y) back to decision variables (X), prioritizing modularity so that new models and validation strategies can be tested without disrupting the core pipeline.

---

## 🏛️ Architectural Style

### 1. Modular Monolith
The codebase is organized into independent modules under `src/modules/`. This allows us to keep the project easy to deploy and manage while maintaining high internal cohesion.

### 2. Domain-Driven Design (DDD)
We use a layered approach to isolate business logic from technical details:
- **Domain**: Pure business rules (e.g., Pareto dominance, feasibility logic).
- **Application**: Use cases that coordinate actions (e.g., "Train a Model").
- **Infrastructure**: Concrete implementations (e.g., PyTorch trainers, JSON storage).

### 3. Ports & Adapters (Hexagonal)
The domain layer defines **Interfaces** (Ports), and the infrastructure layer provides **Implementations** (Adapters). This makes the system "pluggable" — we can swap a scikit-learn model for a PyTorch one by simply changing the adapter.

---

## 🛠️ Layer Responsibilities

| Layer | Responsibility | Key Components |
|-------|----------------|----------------|
| **Domain** | Core logic & abstractions | `InverseEstimator` base, `FeasibilityChecker`, `ParetoFront` |
| **Application** | Orchestration & Use Cases | `TrainModelHandler`, `GenerateDecisionHandler` |
| **Infrastructure** | Technical implementation | `MDNAdapter`, `NPZRepository`, `PlotlyVisualizer` |
| **CLI** | User interaction | `click` commands for training and generation |
| **Workflows** | End-to-end pipelines | `DecisionGenerationWorkflow` |

---

## 🔄 Research-to-Production Path

One of the project's key strengths is its ability to handle different levels of stability:

1.  **Notebooks (`/notebooks`)**: Used for "scratchpad" research, initial data exploration, and messy prototyping.
2.  **Infrastructure Overrides**: Once a prototype works, it's moved into an Infrastructure adapter.
3.  **Core Domain**: Only the most stable, mathematically verified logic resides in the Domain layer.

---

## 🚀 Key Capabilities

- **Hybrid Modeling**: Supports both deterministic regressors and probabilistic generative models.
- **Strict Feasibility Gates**: Implements multi-stage target validation (bounds + proximity).
- **Automated Verification**: Forward-checks proposed designs to verify they match target objectives.
- **Reproducible CLI**: Every research step is captures as a versionable CLI command.

---

## 📈 Future Evolution

The design is intentionally "open-closed":
- **Open for Extension**: New estimators can be added by implementing the `InverseEstimator` interface.
- **Closed for Modification**: Adding a new model doesn't require changing the training orchestrator or the CLI handlers.
