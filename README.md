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

## 🚦 Getting Started (Local)

### Prerequisites
- [Docker](https://www.docker.com/) & [Compose](https://docs.docker.com/compose/)
- [uv](https://github.com/astral-sh/uv) (for local python development)

### Quick Start
```bash
docker compose -f docker-compose.dev.yml up --build
```
- **REST API**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs`

---

## 📄 LICENSE

MIT License - see [LICENSE](LICENSE).

---

Related: [Frontend Repository](https://github.com/Nicola-Ibrahim/Tracing-Objectives-Backwards-Frontend) *(TBD)*
