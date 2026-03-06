from fastapi import APIRouter, Depends, HTTPException

from ....modules.evaluation.application.check_engine_performance import (
    CheckModelPerformanceService,
)
from ....modules.evaluation.application.diagnose_engines import (
    DiagnoseInverseModelsParams,
    DiagnoseInverseModelsService,
    InverseEngineCandidate,
)
from .dependencies import get_diagnose_service, get_performance_service
from .schemas import (
    DiagnoseRequest,
    DiagnoseResponse,
    PerformanceRequest,
    PerformanceResponse,
)

router = APIRouter()


@router.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose_engines(
    request: DiagnoseRequest,
    service: DiagnoseInverseModelsService = Depends(get_diagnose_service),
):
    """
    Run comparative diagnostics across multiple engines.
    Checks cache first to prevent redundant compute.
    """
    try:
        # Map candidates
        candidates = [
            InverseEngineCandidate(solver_type=c.solver_type, version=c.version)
            for c in request.candidates
        ]
        params = DiagnoseInverseModelsParams(
            dataset_name=request.dataset_name,
            inverse_engine_candidates=candidates,
            num_samples=request.num_samples,
            scale_method=request.scale_method,
        )

        # 1. OPTIONAL: Always check cache first in the main endpoint too for robustness
        cached = service.get_cached_diagnostics(params)
        if cached:
            return DiagnoseResponse(**cached)

        # 2. Execute full diagnosis if not cached
        result = service.execute(params)

        return DiagnoseResponse(
            dataset_name=request.dataset_name,
            engines=result["engines"],
            ecdf=result["ecdf"],
            pit=result["pit"],
            mace=result["mace"],
            warnings=result.get("warnings", []),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic failed: {str(e)}")


@router.post("/diagnose/cached", response_model=DiagnoseResponse)
async def get_cached_diagnose(
    request: DiagnoseRequest,
    service: DiagnoseInverseModelsService = Depends(get_diagnose_service),
):
    """
    Retrieve existing diagnostics from database if available.
    Returns 404 if any engine in the request has no cached diagnostics.
    """
    try:
        candidates = [
            InverseEngineCandidate(solver_type=c.solver_type, version=c.version)
            for c in request.candidates
        ]
        params = DiagnoseInverseModelsParams(
            dataset_name=request.dataset_name,
            inverse_engine_candidates=candidates,
            num_samples=request.num_samples,
            scale_method=request.scale_method,
        )

        cached = service.get_cached_diagnostics(params)
        if not cached:
            raise HTTPException(
                status_code=404,
                detail="No cached diagnostics found for requested engines",
            )

        return DiagnoseResponse(**cached)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=404, detail="Cache lookup failed")


@router.post("/performance", response_model=PerformanceResponse)
async def check_performance(
    request: PerformanceRequest,
    service: CheckModelPerformanceService = Depends(get_performance_service),
):
    """
    Check performance of a single engine.
    """
    try:
        result = service.execute(
            dataset_name=request.dataset_name,
            engine=request.engine.model_dump(),
            n_samples=request.n_samples,
        )

        return PerformanceResponse(
            dataset_name=request.dataset_name,
            solver_type=request.engine.solver_type,
            version=result.get("version", 0),
            insights=result.get("insights", {}),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Performance check failed: {str(e)}"
        )
