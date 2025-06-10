# 🧭 Layering Guide: Domain vs Application vs Infrastructure

This guide helps you consistently distinguish where a module, class, or function belongs in a Clean Architecture + DDD-based system.

---

## 🧠 1. The Domain Layer

> "The heart of the system. It knows the business, not the technology."

### ✅ What belongs here

- **Entities**: Core objects with identity and lifecycle (e.g., `User`, `Experiment`, `Run`)
- **Value Objects**: Immutable, comparison-based types (e.g., `Coordinates`, `FeatureRange`)
- **Domain Services**: Pure logic related to the problem domain (e.g., `Normalizer`, `DistanceCalculator`)
- **Business Rules**: Anything that expresses domain logic independent of infrastructure.

### ✅ Allowed Dependencies

- Math libraries (`numpy`, `scipy`)
- Data interfaces like `sklearn.base.BaseEstimator` — **only if used as internal modeling helpers**
- No I/O, no databases, no framework-specific code.

### ❓ Ask

- Is this code part of how we **define and enforce domain rules**?
- Can I run this logic **without a web server, database, or external system**?
- Would a domain expert understand what this code is doing?

---

## ⚙️ 2. The Application Layer (Use Cases)

> "What the system does for the user. It coordinates domain logic."

### ✅ What belongs here

- Use case handlers (`TrainModel`, `EvaluateModel`, `RunSimulation`)
- Orchestration logic (e.g., coordinating components)
- Application-specific interfaces (e.g., repository interfaces, logger interfaces)
- Batch jobs, background tasks, task pipelines.

### ❓ Ask

- Is this coordinating how components interact?
- Does it **depend on the domain**, but not on infrastructure?
- Would changing the UI or DB not affect this logic?

---

## 🧩 3. The Infrastructure Layer (Adapters)

> "Implements technical details. It plugs into the outside world."

### ✅ What belongs here

- Actual database models (`SQLAlchemy`, `MongoEngine`)
- API clients, file readers/writers
- Web frameworks (`FastAPI`, `Flask`, `Django`)
- ML framework adapters, CLI tools, external APIs
- Logging, metrics, and monitoring implementations

### ❓ Ask

- Does this code talk to the **outside world**?
- Does it use framework- or tool-specific APIs?
- If I swapped out the DB, logger, or UI — would this code need to change?

---

## 🧪 Examples

| Module                      | Layer            | Reason |
|-----------------------------|------------------|--------|
| `HypercubeNormalizer`       | **Domain**        | Pure math, no I/O or ML framework code |
| `TrainModelUseCase`         | **Application**   | Coordinates normalizers, model fitting, logging |
| `WandbLogger`               | **Infrastructure**| External logging tool |
| `FastAPI endpoints`         | **Infrastructure**| Web interface adapter |
| `ParetoFrontCalculator`     | **Domain**        | Pure business logic |
| `sklearn.pipeline.Pipeline`| **Infrastructure**| Framework-specific composition and I/O |

---

## ✅ Rule of Thumb

> **If it expresses business logic or rules, it's domain.  
If it wires things together, it's application.  
If it talks to the outside world, it's infrastructure.**

---

## 🧠 Additional Questions to Ask

- Would this code break if I removed the internet? → Maybe Infrastructure  
- Would this code change if I switch web or DB frameworks? → Infrastructure  
- Does this logic depend only on in-memory data and domain rules? → Domain  
- Does this logic coordinate multiple domain actions? → Application  

---
