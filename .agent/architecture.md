# Architecture Context

This repository is a monorepo containing a Next.js frontend and a FastAPI backend.

- `/frontend`: Next.js App Router (React, Tailwind, Recharts). Uses Route Groups `(marketing)` and `(dashboard)`.
- `/backend`: Python FastAPI Modular Monolith. 
  - `/backend/api`: Only contains routing and dependency injection.
  - `/backend/modules`: Contains isolated domain logic (e.g., `users/`, `billing/`). Modules must not directly import from each other's infrastructure layers.