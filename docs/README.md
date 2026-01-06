# 🧭 Developer Portal

Welcome to the internal documentation for **Tracing Objectives Backwards**. This portal is designed to help you navigate the codebase, understand the underlying theory, and contribute effectively.

---

## 🚀 Getting Started

- **[Installation & Setup](guide/usage.md)**: How to get the project running locally using `uv`.
- **[CLI Reference](guide/usage.md#direct-cli-entry-points)**: Common commands for training, evaluation, and visualization.
- **[Building & Structure](guide/building.md)**: Deep dive into the project's build system and directory layout.

---

## 🧠 Conceptual Guides

- **[Inverse Design Pipeline](processes/inverse-design-pipeline.md)**: The "Heart" of the project – how we map objectives back to decisions.
- **[DDD & Architecture](concepts/ddd-architecture-guide.md)**: Understanding our layers (Domain, Application, Infrastructure).
- **[Feasibility Methods](concepts/feasibility-methods.md)**: How we determine if a target objective is reachable.
- **[Model Training & Validation](processes/model-training-validation.md)**: Standardized workflows for model evaluation.

---

## 🔬 Modeling & Estimators

Documentation for the specific models implemented in the zoo:

- **[Mixture Density Networks (MDN)](modeling/mdn.md)**
- **[Conditional VAE (CVAE)](modeling/conditional-vae.md)**
- **[Invertible Neural Networks (INN)](modeling/inn.md)**
- **[NSGA-II Optimization](modeling/nsga2-optimization.md)**

---

## 📑 Specifications & Technical Docs

- **[System Design Overview](specs/design.md)**: High-level architectural decisions and rationale.
- **[System Framework](processes/system-framework.md)**: Technical specifications of the processing framework.
- **[Requirements](specs/requirements.md)**: Core project dependencies and constraints.

---

## 🛠 Project Management

- **[Active Tasks](specs/tasks.md)**: Current roadmap and outstanding items.
