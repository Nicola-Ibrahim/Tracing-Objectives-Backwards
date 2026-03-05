# Feature Specification: Evaluation Engine Refinement & Analytics Enhancement

**Feature Branch**: `028-fix-eval-services`  
**Created**: 2026-03-05  
**Status**: Draft  
**Input**: User description: "enable Evaluation page and ensure linking with the api endpoints, fix backend diagnose and compare services to return correct data, and ensure beautiful frontend graphs/charts"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Reliable Comparative Diagnosis (Priority: P1)

As a researcher, I want the system to successfully run diagnostics across multiple engines so that I can compare their calibration (ECDF, PIT) and accuracy (MACE) without data parsing errors or service failures.

**Why this priority**: Core value of the platform is evaluating inverse engines. Current service failures block this insight.

**Independent Test**: Trigger a "Comparative Diagnosis" from the Evaluation page selecting 2+ engines. The system should return result charts without crashing.

**Acceptance Scenarios**:

1. **Given** 2 trained engines for the same dataset, **When** clicking "Run Comparative Diagnosis", **Then** ECDF and PIT charts appear with lines for both engines.
2. **Given** a successful diagnosis, **When** viewing the MACE table, **Then** all selected engines are listed with their corresponding scores.

---

### User Story 2 - High-Fidelity Analytics Visualization (Priority: P1)

As a user, I want the graphs and charts in the Evaluation workspace to be "beautiful" (using vibrant colors, smooth transitions, and premium aesthetics) so that I can easily distinguish between engine performances and enjoy a premium UI.

**Why this priority**: User specifically requested a "beautiful display" to improve UX and professionalism.

**Independent Test**: Open the Evaluation page and verify chart aesthetics (gradients, hover effects, line smoothing).

**Acceptance Scenarios**:

1. **Given** diagnostic results, **When** hovering over a line in the ECDF chart, **Then** a beautiful, informative tooltip appears and other lines subtly dim.
2. **Given** multiple engines, **When** charts render, **Then** they use a vibrant, distinct color palette suitable for high-end analytics.

---

### User Story 3 - Robust Multi-Engine Generation (Priority: P2)

As a developer, I want the "Candidate Generation" (compare) service to return serialized data correctly so that I can visualize generated candidates across different solvers without backend 500 errors.

**Why this priority**: Essential for verifying engine behavior on specific target points.

**Independent Test**: Run "Generate Candidates" for a specific objective using 2+ engines and verify the scatter plots render.

**Acceptance Scenarios**:

1. **Given** a target objective, **When** generating candidates, **Then** the backend returns a list of candidate points (not numpy arrays) that are successfully parsed by the frontend.

---

### Edge Cases

- **Mixed Version Compatibility**: How does the system handle comparing older versions of engines with newer ones? (Requirement: Support version-aware identifiers).
- **Empty Result Sets**: What if an engine fails to generate any valid candidates? (Requirement: Graceful degradation in charts).
- **Mismatched Dimensionality**: Attempting to compare engines trained on different datasets? (Requirement: Filter available engines by dataset).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Backend MUST align `diagnose` output structure to match frontend expectations (`x` array + `y` record dictionary).
- **FR-002**: Backend MUST calculate a shared X-axis for ECDF and PIT charts to enable side-by-side comparison on the same coordinates.
- **FR-003**: Backend MUST serialize all numpy outputs (candidates, distances, scores) to standard Python lists/floats before API response.
- **FR-004**: Frontend `Charts.tsx` MUST implement enhanced styling (vibrant color tokens, glassmorphic card containers, animated data entry).
- **FR-005**: Frontend MUST handle empty or single-engine diagnostic states gracefully without layout breakages.
- **FR-006**: Backend `DiagnoseInverseModelsService` MUST verify that requested engine versions actually exist before attempting compute.

### Key Entities *(include if feature involves data)*

- **DiagnosticResult**: Represents the outcome of an audit (Accuracy/Reliability) for a specific engine/dataset pair.
- **MetricPlotData**: The UI-facing value object containing coordinate arrays for plotting.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: API response time for cached diagnostics is under 300ms.
- **SC-002**: 100% of "Run Diagnosis" attempts for existing engines result in visible, correctly formatted charts.
- **SC-003**: Charts support 5+ simultaneous engines without losing legibility.
- **SC-004**: Frontend passes accessibility contrast checks for all chart color palettes.
