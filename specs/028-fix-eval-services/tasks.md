---
description: "Implementation tasks for Evaluation Engine Refinement"
---

# Tasks: Evaluation Engine Refinement & Analytics Enhancement

**Input**: Design documents from `/specs/028-fix-eval-services/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/api.md

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure.

Currently, no setup tasks are required as this is an inline fix on existing modules.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented.

- [ ] T001 Update `MetricPlotData` interface in `frontend/src/features/evaluation/types.ts`
- [ ] T002 Update `DiagnoseResponse` schema in `src/api/v1/evaluation/schemas.py`

**Checkpoint**: Shared data contracts are established.

---

## Phase 3: User Story 1 - Reliable Comparative Diagnosis (Priority: P1) 🎯 MVP

**Goal**: Ensure the backend `diagnose` service successfully outputs non-interpolated, JSON-serializable `{x, y}` series for ECDF/PIT curves, allowing the frontend to render multiple engines without crashing.

**Independent Test**: Trigger a "Comparative Diagnosis" on the Evaluation page for multiple engines and verify charts render without frontend/backend crash.

### Implementation for User Story 1

- [ ] T003 [US1] Refactor `_calculate_ecdf` and output formatting in `src/modules/evaluation/application/diagnose_models.py`
- [ ] T004 [US1] Refactor `PerformanceChart` in `frontend/src/features/evaluation/components/Charts.tsx` to handle independent `{x, y}` data traces per engine
- [ ] T005 [P] [US1] Update `EngineComparisonPanel.tsx` or `evaluation/page.tsx` if data passing logic requires alignment (verify)

**Checkpoint**: At this point, comparative diagnostics will functionally render in the charts.

---

## Phase 4: User Story 2 - High-Fidelity Analytics Visualization (Priority: P1)

**Goal**: Upgrade the UI of the Evaluation workspace's charts to be "beautiful" with vibrant colors, smooth line tension, glassmorphism, and better tooltips.

**Independent Test**: Visually inspect the Evaluation page charts for improved aesthetics and interact with the tooltips.

### Implementation for User Story 2

- [ ] T006 [P] [US2] Apply premium color palettes and styling in `frontend/src/features/evaluation/components/Charts.tsx`
- [ ] T007 [P] [US2] Add glassmorphic containers and subtle animations to the evaluation dashboard layout in `frontend/src/app/(dashboard)/evaluation/page.tsx`

**Checkpoint**: Evaluation dashboard is functionally robust and visually stunning.

---

## Phase 5: User Story 3 - Robust Multi-Engine Generation (Priority: P2)

**Goal**: Ensure the "Generate Candidates" (compare) service properly serializes all numpy arrays so the API returns clean JSON instead of 500 errors.

**Independent Test**: Use the "Generate Candidates" feature with multiple engines to verify no server parsing errors occur and the candidate scatter plot is drawn.

### Implementation for User Story 3

- [ ] T008 [US3] Explicitly convert numpy types to native Python types in `InverseModelsCandidatesComparator` (`src/modules/evaluation/application/inverse_model_candidates_comparator.py`)
- [ ] T009 [US3] Ensure `results_map` builds purely serializable dictionaries in `CompareInverseModelCandidatesService` (`src/modules/evaluation/application/compare_candidates.py`)

**Checkpoint**: All multi-engine simulation and generation workflows operate correctly.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T010 Run end-to-end manual verification using the `quickstart.md` guide.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Foundational (Phase 2)**: BLOCKS User Story 1
- **User Story 1 (Phase 3)**: MUST be complete before User Story 2 (styling). 
- **User Story 2 (Phase 4)**: Safe to execute after US1.
- **User Story 3 (Phase 5)**: Can be executed in parallel with US1/US2 as it touches a different backend service.

### Parallel Opportunities

- T006 and T007 can be executed in parallel.
- The entirety of US3 (Phase 5) can be executed in parallel with US1 (Phase 3) backend fixes.

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 2: Foundational interface changes
2. Complete Phase 3: Update `Charts.tsx` logic and `diagnose_models.py` return structures.
3. Validate independent {x, y} rendering.

### Incremental Delivery

1. Foundation + US1 -> Diagnoses work mathematically and render safely.
2. Add US2 -> The charts become premium and beautiful.
3. Add US3 -> The Candidate Generator feature is stabilized for multi-engine use.
