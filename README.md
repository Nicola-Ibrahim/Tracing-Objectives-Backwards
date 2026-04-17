# 🎯 Tracing Objectives Backwards (Backend & Infrastructure)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Infrastructure-Docker-blue.svg)](docker-compose.prod.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**High-Performance AI Orchestrator for Multi-Objective Optimization.**

This repository contains the mathematical engine, REST API, and deployment infrastructure for **Tracing Objectives Backwards**. 

---

## 🏗️ System Role

This serves as the central anchor for:
1. **The Engine**: High-concurrency AI orchestrator handling data simulation, model training, and generative diagnostics.
2. **Infrastructure Gateway**: Nginx and Docker configurations for VPS-level deployment.
3. **API Documentation**: Automated OpenAPI/Swagger schemas.

---

## 🏛️ Architecture Overview

The system is designed as a **modular domain monolith** using FastAPI.
- **`backend/`**: Core API and mathematical logic.
- **`infra/`**: Nginx configuration and SSL management.
- **`docker-compose.*.yml`**: Production and development orchestration.

---

## 🚦 Development Workflow

To streamline development, we use three core automation scripts.

### 1. One-Time Setup
Run this once when you first clone the repository or on a new machine. It installs all necessary tools (`uv`, `pnpm`, `doppler`, etc.) and ensures your path is configured.
```bash
python3 scripts/bootstrap.py
```

### 2. Daily Start (The Morning Routine) ☕
Run this every morning to get your environment ready. It syncs dependencies, boots containers, and automatically launches your dev servers in new terminal tabs.
```bash
python3 scripts/setup_dev.py
```

### 3. Reset Environment
Run this when you switch branches or if your local state is out of sync. It refreshes secrets from Doppler and flushes the Redis cache.
```bash
python3 scripts/init_env.py
```

---

## 🛠️ Infrastructure Overview

- **REST API**: `http://localhost:8000` (FastAPI)
- **Frontend**: `http://localhost:3000` (Next.js)
- **Interactive Docs**: `http://localhost:8000/docs`
- **Secret Management**: Powered by [Doppler](https://www.doppler.com/)
- **Process Orchestration**: Docker Compose (`redis`, `nginx`)

---

## 📄 LICENSE

MIT License - see [LICENSE](LICENSE).

---

Related: [Frontend Repository](https://github.com/Nicola-Ibrahim/Tracing-Objectives-Backwards-Frontend) *(TBD)*



# 🚀 Backend: Inverse Mapping Engine

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Infrastructure-Docker-blue.svg)](Dockerfile)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

This is the high-performance AI engine for **Tracing Objectives Backwards**. Built with **FastAPI** and **PyTorch**, it provides the mathematical foundation for inverse exploration, training sophisticated surrogates, and auditing their reliability.

---

## 🏛️ System Architecture (C4 Container View)

The backend is structured as a **Modular Monolith** following **Clean Architecture**. The API layer acts as a thin orchestrator for the core domain modules.

```mermaid
graph TD
    User([Researcher/Engineer])
    
    subgraph "Tracing Objectives Backwards System"
        subgraph "Backend Container (FastAPI)"
            API["🛣️ API Layer<br/>(FastAPI Containers)"]
            
            subgraph "Domain Modules (Bounded Contexts)"
                Dataset["📊 Dataset<br/>(Gen & Normalization)"]
                Modeling["🧠 Modeling<br/>(Train & Predict)"]
                Inverse["🔄 Inverse<br/>(Synthesis & Propose)"]
                Evaluation["🔬 Evaluation<br/>(Audit & Diagnose)"]
            end
        end
        
        FileSystem[("📂 Data Store<br/>(Local Artifacts)")]
    end

    User -->|Uses| API
    API -->|Orchestrates| Dataset
    API -->|Orchestrates| Modeling
    API -->|Orchestrates| Inverse
    API -->|Orchestrates| Evaluation

    Dataset -->|Saves/Reads| FileSystem
    Modeling -->|Saves/Reads Models| FileSystem
```

---

## 🧬 Core Domain Modules

Each module is an isolated **Bounded Context** with its own Domain and Infrastructure layers.

- **`📊 dataset`**: Handles simulation of raw Pareto-optimal data. It manages the lifecycle of ground-truth datasets and ensures consistent normalization for AI training.
- **`🧠 modeling`**: The heart of the surrogate engine. It contains implementations for **MDNs (Mixture Density Networks)**, **CVAEs**, and the custom **GPBI** algorithm. We chose PyTorch for its dynamic computational graphs, essential for custom loss functions.
- **`🔄 inverse`**: Implements the synthesis logic. It uses trained models to propose design candidates (X) that match target objectives (Y), handling the one-to-many mapping challenge.
- **`🔬 evaluation`**: The "Auditor". It runs comprehensive diagnostics like **PIT (Probability Integral Transform)**, **MACE**, and Diversity audits to ensure model trustworthiness.

---

## 🛠️ Technical Specifications

### 🚦 Quick Start (Local)

#### Prerequisites
- [uv](https://github.com/astral-sh/uv) (Recommended) or `pip`

#### Installation
```bash
uv sync
```

#### Development Server
```bash
uv run fastapi dev src/api/main.py
```

### 🐳 Dockerization

To run the backend as a standalone container:

**1. Build the image:**
```bash
docker build -t tob-backend .
```

**2. Run the container:**
```bash
docker run -p 8000:8000 tob-backend
```

### 🧪 CLI Tools (Poe)
We use `poethepoet` to manage common development tasks:
- `uv run poe train-inv`: Launch inverse model training.
- `uv run poe diagnose`: Run the full auditor suite on trained models.
- `uv run poe test`: Execute the pytest-based test suite.

---

## 📖 Extended Knowledge

For a deeper look into the patterns and math:
- 🏛️ **[Central DDD Guide](../docs/concepts/ddd-architecture-guide.md)**
- 🧬 **[Inverse Design Theory](../docs/processes/inverse-design-pipeline.md)**
- 🧭 **[Developer Portal](../docs/README.md)**
- 📖 **[Technical Glossary](docs/GLOSSARY.md)**

---
Related: [Root README](../README.md) | [Frontend README](../frontend/README.md)
