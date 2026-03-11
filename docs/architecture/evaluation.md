← [Back to Overview](README.md)

# ⚖️ Evaluation Module

**Bounded Context**: Generative Diagnostics  
**Aggregate Root**: `DiagnosticResult`

The `evaluation` module is the most complex bounded context in the system. It contains three distinct sub-contexts: **core diagnostics** (accuracy and reliability of surrogates), **decision validation** (out-of-distribution detection), and **feasibility** (constraint satisfaction and Pareto proximity).

## 🥞 DDD Architecture & Sub-Contexts

```mermaid
flowchart LR
    subgraph Core ["Core Diagnostics (Accuracy & Reliability)"]
        direction TB
        E[DiagnosticResult]
        L[AccuracyLens<br/>ReliabilityLens]
        S[GenerativeDistributionAuditor<br/>SpatialCandidateAuditor<br/>Scaling]
    end

    subgraph Validation ["decision_validation Sub-Context"]
        direction TB
        VE[DecisionValidationCalibration<br/>GeneratedDecisionValidationReport]
        VV[Verdict, GateResult]
        VI[BaseValidator<br/>BaseCalibratorRepository]
    end

    subgraph Feasibility ["feasibility Sub-Context"]
        direction TB
        FE[FeasibilityAssessment<br/>AssessmentFinding]
        FV[ParetoFront, Score, Tolerance, Suggestions]
        FS[ObjectiveFeasibilityService]
        FI[Diversity, Scoring]
    end
    
    subgraph Application ["Application Layer"]
        direction TB
        U[diagnose_models<br/>compare_candidates<br/>visualize_diagnostics<br/>check_performance]
        X[train_grid_search*]
    end
    
    Application --> Core
    Application --> Validation
    Application --> Feasibility
    
    style X fill:#ff7675,color:#2d3436,stroke:#d63031,stroke-width:2px;
```

> **\* Architectural Debt Alert**: `train_grid_search` physically resides in the `evaluation/application/use_cases/` directory but semantically belongs to the `modeling` context, as it trains and persists estimators. See [Integration Debt](integration.md#architectural-debt).

## 📦 Component Inventory

### Core Diagnostics

| Layer | Type | Component | Description |
|-------|------|-----------|-------------|
| **Domain** | Entity | `DiagnosticResult` | Aggregate root storing the complete audit of an estimator for a dataset. |
| **Domain** | Entity | `AccuracyLens` | Objective-space spatial discrepancy metrics (Euclidean distance, best-shot). |
| **Domain** | Entity | `ReliabilityLens` | Decision-space distribution metrics (PIT, CRPS, Calibration Error). |
| **Domain** | Service | `SpatialCandidateAuditor` | Computes distances between generated candidates and reference targets. |
| **Domain** | Service | `GenerativeDistributionAuditor` | Computes statistical metrics comparing generated distributions vs ground truth. |
| **App** | Use Case | `diagnose_models` | End-to-end pipeline running full suite of metrics on inverse models. |
| **App** | Use Case | `compare_candidates` | Generates candidates across multiple inverse models and visualizes comparison. |

### Decision Validation Sub-Context

| Layer | Type | Component | Description |
|-------|------|-----------|-------------|
| **Domain** | Entity | `DecisionValidationCalibration` | Calibration thresholds (e.g., Mahalanobis distance cutoffs). |
| **Domain** | Entity | `GeneratedDecisionValidationReport` | Results of validating AI-generated decisions. |
| **Domain** | Enum | `Verdict` | `ACCEPT` \| `REJECT` \| `WARNING` \| `ABSTAIN` |
| **Domain** | Interface | `BaseValidator` | Contract for OOD detection mechanisms. |
| **Infra** | Validator | Mahalanobis, Split Conformal L2 | Concrete out-of-distribution detectors. |

### Feasibility Sub-Context

| Layer | Type | Component | Description |
|-------|------|-----------|-------------|
| **Domain** | Entity | `FeasibilityAssessment` | Overall rating of whether an objective is feasible. |
| **Domain** | Entity | `AssessmentFinding` | Specific warnings/failures (e.g., "Outside historical range"). |
| **Domain** | Value | `ParetoFront`, `Suggestions` | Derived constraint boundaries and fallback ideas. |
| **Domain** | Service | `ObjectiveFeasibilityService` | Runs target objectives through policy validators. |
| **Domain** | Policy | `HistoricalRangeValidator` | Asserts objectives don't violate known absolute min/max bounds. |
| **Domain** | Policy | `ParetoProximityValidator` | Asserts objectives aren't utopic (beyond the Pareto front). |

## 🔄 Diagnostic Service Flow

```mermaid
flowchart TD
    A[Load Dataset & Forward Estimator] --> B[Load Inverse Estimator Candidates]
    B --> C[Sample X from Inverse Estimator]
    
    C --> D[GenerativeDistributionAuditor]
    D -->|PIT, CRPS, Calibration| E[ReliabilityLens]
    
    C --> F[Forward-Predict Y from sampled X]
    F --> G[SpatialCandidateAuditor]
    G -->|Discrepancy, Bias, Dispersion| H[AccuracyLens]
    
    E --> I[Assemble DiagnosticResult Aggregate]
    H --> I
    I --> J[Save to DiagnosticRepository]
```

---
Related: [modeling](modeling.md) | [integration](integration.md)
