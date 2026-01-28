# 🧭 Domain-Driven Design Layering Guide

This guide helps you apply **Domain-Driven Design (DDD)** and **Clean Architecture** in this project by clearly distinguishing **what goes where**.

---

## 📚 Overview of Architectural Layers

| Layer | Responsibility | Knows About | Doesn't Know About | Project Example |
|-------|----------------|-------------|---------------------|-----------------|
| **Domain** | Core mapping rules, business logic | Itself | I/O, CLI, ML Libs | `src/modules/optimization_engine/domain/services/inverse_validator.py` |
| **Application**| Coordinating data & models | Domain | UI, DB internals | `src/modules/optimization_engine/application/use_cases/train_inverse_model.py` |
| **Infrastructure**| External tools & I/O | Everything | (Lowest layer) | `src/modules/optimization_engine/infrastructure/modeling/adapters/mdn.py` |

---

## 🧠 1. Domain Layer – “The Heart”

> "The heart of the system. It knows the inverse design theory, not the technology."

### ✅ What belongs here
- **Base Interfaces**: `BaseInverseEstimator`, `BaseRepository`.
- **Domain Services**: `InverseModelValidator`, `FeasibilityChecker`.
- **Entities & Value Objects**: `Point`, `Bounds`, `DatasetMetadata`.

**Example:**
The logic for checking if a target objective is "close enough" to the Pareto front is a domain rule. It shouldn't care if the data comes from a JSON file or a database.

---

## ⚙️ 2. Application Layer – “The Orchestrator”

> "The 'glue' that coordinates domain logic to serve a user's goal."

### ✅ What belongs here
- **Command Handlers**: `TrainInverseModelHandler`, `GenerateDecisionHandler`.
- **Port Interfaces**: Definitions for how we log or plot.

**Example:**
A handler that pulls a dataset from a repository, feeds it to an estimator for training, and then logs the results to the dashboard.

---

## 🧩 3. Infrastructure Layer – “The Implementation”

> "Implements the technical details that change most often."

### ✅ What belongs here
- **Model Adapters**: `PytorchCVAEAdapter`, `SklearnRBFAdapter`.
- **Repositories**: `NPZDatasetRepository`.
- **Visualizers**: `PlotlyDiagnosticVisualizer`.

**Example:**
The actual code that calls `torch.nn.Module` or `sklearn.fit()` lives here. If we switch from PyTorch to JAX, we only change this layer.

---

## 🧱 Project Directory Mapping

```plaintext
src/modules/optimization_engine/
├── domain/
│   ├── services/    # e.g., inverse_validator.py
│   └── entities/    # e.g., dataset_metadata.py
├── application/
│   ├── use_cases/   # e.g., train_inverse_model.py
│   └── handlers/    # e.g., train_handler.py
├── infrastructure/
│   ├── modeling/    # e.g., mdn_adapter.py
│   └── repositories/# e.g., npz_repository.py
└── cli/             # e.g., train_command.py
```

---

## ✅ Rule of Thumb

> 🟢 **If it expresses the 'Math' or 'Rules' of inverse design, it's Domain.**  
> 🟡 **If it 'Coordinates' multiple steps to achieve a task, it's Application.**  
> 🔴 **If it imports a 'Library' like Torch, Sklearn, or Plotly, it's Infrastructure.**

---

## 💡 Final Thought

> **"Code should scream the domain."** — Eric Evans  
Structure your code so its intent is obvious. When you look at `src/`, you should see "Optimization Engine" and "Inverse Mapping", not just "Python scripts".
