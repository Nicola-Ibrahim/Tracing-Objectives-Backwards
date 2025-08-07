
# 📝 Core Tasks for AI Inverse Mapping System

## Main System Process (Mermaid Diagram)

```mermaid
flowchart TD
    A[User specifies target objective Y*] --> B[Normalize Y*]
    B --> C[Feasibility Check (LSNF)]
    C -->|Feasible| D[Predict X* via Inverse Interpolator]
    D --> E[Denormalize X*]
    E --> F[Evaluate f(X*)]
    F --> G[Compute Error Metrics]
    C -->|Infeasible| H[Suggest Feasible Alternatives]
    H --> I[Repeat with New Y*]
```

---

## Core Development Tasks

- **Domain Modeling:**
  - Implement domain entities, value objects, aggregates, and services for optimization and mapping.
  - Define clear boundaries and invariants for aggregates.
- **Model Development:**
  - Develop and train inverse decision mapper models (deterministic: Clough-Tocher, Kriging, Linear, NN, RBF, Spline, SVR; probabilistic: Gaussian Process, Conditional VAE).
  - Implement model selection and evaluation strategies.
- **Feasibility Checking:**
  - Integrate Local Spherical Neighborhood Feasibility (LSNF) and other validation methods.
  - Develop scoring and diversity strategies for feasible suggestions.
- **Pipeline Construction:**
  - Build the end-to-end pipeline for inverse mapping (Y → X), including normalization, prediction, denormalization, and evaluation.
  - Ensure modularity and extensibility for new models and validation strategies.
- **Visualization Infrastructure:**
  - Create interactive visualizations for metrics and results (e.g., Plotly box, violin, bar plots).
  - Integrate dashboards for model comparison and error analysis.
- **Testing & Validation:**
  - Validate and test the system with real and synthetic data.
  - Implement automated tests for core components and workflows.
- **Documentation & Diagrams:**
  - Document architecture, workflows, and domain concepts.
  - Maintain and update diagrams (C4, process, model architecture).

---

## Additional Tasks

- **Model Expansion:**
  - Add support for new models (e.g., Conditional VAE, NSGA-II, other generative models).
  - Benchmark and compare model performance.
- **Visualization Enhancement:**
  - Enhance visualization and reporting capabilities.
  - Add new plot types and interactive features.
- **Domain Layer Refinement:**
  - Refine domain layer for aggregate boundaries, value objects, and domain events.
  - Ensure business logic is isolated from technical concerns.
- **Documentation Growth:**
  - Expand catalog of terms and update documentation as system evolves.
  - Add user guides and API references.
- **Collaboration & Review:**
  - Conduct code reviews and collaborative design sessions.
  - Gather user feedback for continuous improvement.

---

## Additional Information

- **Key Technologies:** Python, Plotly, scikit-learn, PyTorch/TensorFlow (for neural models), Mermaid (for diagrams).
- **Data Sources:** Historical Pareto-optimal datasets, synthetic benchmarks.
- **Metrics:** Mean Squared Error (MSE), Mean Absolute Error (MAE), feasibility scores, diversity metrics.
- **Extensibility:** System is designed for easy integration of new models, validation strategies, and visualization tools.
