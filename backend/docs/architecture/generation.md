← [Back to Overview](README.md)

# 🎯 Generation Module

**Bounded Context**: Coherent Candidate Synthesis  
**Aggregate Root**: `CoherenceContext`

The `generation` module is a dedicated pipeline for producing physically valid decision variables given a target objective. Instead of relying solely on global Inverse Surrogates (which often struggle with non-uniqueness and strict physical constraints), this module uses a localized approach combining geometric meshing and surrogate-assisted optimization.

## 🏗️ Architectural Pattern

This module follows the **Clean Architecture** patterns defined in our **[DDD Guide](../concepts/ddd-architecture-guide.md)**.

### Layer Mapping
- **Domain**: Geometric localization and coherence rules (`CoherenceContext`, `BarycentricLocator`).
- **Application**: Candidate synthesis use cases.
- **Infrastructure**: Trust-region optimizers.

## 📦 Component Inventory

| Layer | Type | Component | Description |
|-------|------|-----------|-------------|
| **Domain** | Entity | `CoherenceContext` | Aggregate root holding the Delaunay mesh, objective vertices, normalized anchors, and $\tau$ threshold. |
| **Domain** | Value | `CoherenceParams` | Tuning parameters: Dirichlet concentration $\alpha$, number of samples, trust region radius, etc. |
| **Domain** | Value | `GenerationResult` | Final output packaging the generated decisions, their predicted objectives, residual errors, and path taken. |
| **Domain** | Service | `BarycentricLocator` | Finds the mesh simplex containing the target and computes barycentric weights. |
| **Domain** | Service | `CoherenceGate` | Determines if the anchors of a simplex are geometrically tight enough to safely interpolate (using $\tau$). |
| **Domain** | Service | `DirichletSampler` | Samples valid weight variations around the exact target using a Dirichlet distribution constraint. |
| **Domain** | Service | `CandidateRanker` | Evaluates generated candidates through a forward surrogate and ranks them by residual error. |
| **App** | Use Case | `PrepareContextService` | Offline phase: Generates the mesh and computes the safe interpolation threshold $\tau$. |
| **App** | Use Case | `Generate...Service` | Real-time phase: Executes the full localization, gating, and generation pipeline. |
| **Infra** | Optimizer | `TrustRegionOptimizer` | Fallback surrogate-assisted optimizer using `scipy.optimize.minimize(method='trust-constr')`. |
| **Infra** | Repo | `FileSystemContextRepository` | Persists the context arrays via NPZ and metadata via TOML. |

## 🔄 Generation Pipeline Phases

```mermaid
flowchart TD
    %% Phase 0
    subgraph P0 ["Phase 0: Context"]
        A[Load CoherenceContext<br/>(Delaunay Mesh + &tau;)]
    end

    %% Phase 1
    subgraph P1 ["Phase 1: Localization"]
        B[BarycentricLocator]
    end

    %% Phase 2
    subgraph P2 ["Phase 2: Gate"]
        C{Is Target Inside Mesh?}
        D[CoherenceGate]
        E{Is Simplex Coherent<br/>Distance &le; &tau; ?}
    end

    %% Phase 3
    subgraph P3 ["Phase 3: Generation"]
        F[DirichletSampler<br/>(Interpolation Pathway)]
        G[TrustRegionOptimizer<br/>(Extrapolation Pathway)]
        H[Reconstruct Decisions<br/>from Weights]
    end

    %% Phase 4
    subgraph P4 ["Phase 4: Rank"]
        I[Evaluate via Forward Surrogate]
        J[CandidateRanker]
        K((GenerationResult))
    end

    A -->|Target Objective| B
    B --> C
    
    C -->|Yes: Has Anchors| D
    C -->|No: Outside Mesh| G
    
    D --> E
    E -->|Yes: Coherent| F
    E -->|No: Incoherent| G
    
    F --> H
    H --> I
    G --> I
    
    I --> J
    J --> K
    
    classDef path1 fill:#b8e994,stroke:#78e08f,color:#000
    classDef path2 fill:#ffb8b8,stroke:#ff7979,color:#000
    class F,H path1
    class G path2
```

---
Related: [dataset](dataset.md) | [modeling](modeling.md)
