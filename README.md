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
