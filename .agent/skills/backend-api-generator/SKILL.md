---
name: backend-api-generator
description: Scaffolds robust backend APIs using Python, FastAPI, and PostgreSQL. Enforces a Modular Monolith architecture, separating the API routing layer from isolated domain modules.
---
# Backend API Generator Skill

Your goal is to build a highly cohesive, loosely coupled Modular Monolith using Python, FastAPI, and Domain-Driven Design (DDD).

## Core Principles
1. **Modular Monolith Structure:** The application must be split into strictly isolated feature modules (e.g., `modules/users/`, `modules/billing/`). Each module must contain its own Domain, Use Cases, and Infrastructure (repositories).
2. **API vs. Backend Split:** Maintain a clear separation between the API entrypoints and the backend logic. The `api/` folder should only contain FastAPI routers, dependency injection, and HTTP-specific logic. The routers simply call Use Cases from the `modules/` folder.
3. **Module Isolation:** Modules must not directly query another module's database tables or internal repositories. Cross-module communication must happen via public interfaces or domain events.
4. **PostgreSQL & Clean Architecture:** Use async database drivers and map ORM models to Pydantic schemas before returning data to the API layer.

## Execution Workflow
1. **Design the Domain:** Inside the target module (e.g., `modules/inventory/domain/`), define the entities and abstract repository interfaces.
2. **Write the Use Case:** Create the application logic (`modules/inventory/use_cases/`) that orchestrates the domain.
3. **Implement Infrastructure:** Write the concrete PostgreSQL repository (`modules/inventory/infrastructure/`).
4. **Wire the API:** In the separated API folder (`api/v1/endpoints/`), create the FastAPI route that injects the repository into the Use Case and returns the Pydantic response.

## Constraints
* Do not allow one domain module to import internal infrastructure from another module.
* Do not put business logic or direct database queries inside the `api/` routing folders.
* Never write raw, unparameterized SQL queries.