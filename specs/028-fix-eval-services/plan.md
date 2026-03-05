# Implementation Plan: Evaluation Engine Refinement

**Branch**: `028-fix-eval-services` | **Date**: 2026-03-05 | **Spec**: [file:///Users/nicolaibrahim/Desktop/proj/Tracing-Objectives-Backwards/specs/028-fix-eval-services/spec.md]
**Input**: Feature specification from `/specs/028-fix-eval-services/spec.md`

## Summary

Resolves data integration failures between the backend evaluation/compare services and the frontend charts. The backend outputs will be strictly serialized to JSON-compatible lists, and the frontend `Charts.tsx` component will be refactored to consume fully independent `{x, y}` data series per engine, ensuring mathematically sound data visualization. Furthermore, the frontend views will receive styling enhancements to meet the requested "beautiful" criteria.

## Technical Context

**Language/Version**: Python 3.11, React (Next.js)
**Primary Dependencies**: FastAPI, TanStack Query, react-chartjs-2, Chart.js, Shadcn UI
**Target Platform**: Linux backend, Browser frontend
**Project Type**: ML Evaluation Dashboard
**Constraints**: JSON serialization limits for numpy objects, Chart.js strictly typed configurations.
**Scale/Scope**: Updating API routes, application services, and singular frontend view.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Test-First**: Will ensure basic validation of endpoint returns.
- **Observability**: Services are already instrumented with `_logger.log_info`.
- **Simplicity**: Modifying the frontend to accept backend coordinate pairs is simpler and truer to the data than forcing backend interpolation to a shared grid.

## Project Structure

### Documentation (this feature)

```text
specs/028-fix-eval-services/
├── plan.md              # This file
├── research.md          # Strategy for charting logic and serialization
├── data-model.md        # Updated frontend MetricPlotData / backend Result
├── quickstart.md        # Guide to run fixed endpoints
├── contracts/           
│   └── api.md           # API JSON shape adjustments
└── tasks.md             # Execution tracker
```

### Source Code

```text
backend/
└── src/
    └── modules/
        └── evaluation/
            └── application/
                ├── compare_candidates.py           # Add .tolist() serialization
                └── inverse_model_candidates_comparator.py # Float casts

frontend/
└── src/
    └── features/
        └── evaluation/
            ├── types.ts              # Modify MetricPlotData type
            └── components/
                └── Charts.tsx        # Update mapping for independent x/y datasets
```

**Structure Decision**: The changes are localized fixes bridging the `evaluation` application layer boundary and the corresponding frontend feature module.

## Complexity Tracking

No constitution violations detected. The approach focuses on type-safety and correct interface boundaries rather than adding new architectural complexity.
