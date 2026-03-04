from fastapi import APIRouter, Depends, HTTPException

from ....modules.evaluation.application.check_performance import (
    CheckModelPerformanceService,
)
from ....modules.evaluation.application.diagnose_models import (
    DiagnoseInverseModelsService,
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
    """
    try:
        candidates = [c.model_dump() for c in request.candidates]

        result = service.execute(
            dataset_name=request.dataset_name,
            candidates=candidates,
            num_samples=request.num_samples,
            scale_method=request.scale_method,
        )

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
