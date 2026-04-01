from typing import Annotated

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from src.containers import RootContainer
from src.modules.evaluation.application.check_engine_performance import (
    CheckModelPerformanceParams,
    CheckModelPerformanceService,
)
from src.modules.evaluation.application.diagnose_engines import (
    RunDiagnosticsService,
)

from .schemas import (
    DiagnoseAsyncResponse,
    DiagnoseRequest,
    PerformanceRequest,
    PerformanceResponse,
)

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


@router.post("/diagnose", response_model=DiagnoseAsyncResponse)
@inject
async def diagnose_engines(
    request: DiagnoseRequest,
    service: Annotated[
        RunDiagnosticsService,
        Depends(Provide[RootContainer.evaluation.run_diagnostics_service]),
    ],
):
    task_id = await service.execute_async(request)
    return DiagnoseAsyncResponse(task_id=task_id)


@router.get("/status/{task_id}")
@inject
async def task_status_stream(
    task_id: str,
    service: Annotated[
        RunDiagnosticsService,
        Depends(Provide[RootContainer.evaluation.run_diagnostics_service]),
    ],
):
    """
    SSE endpoint to stream task progress from the Service.
    """
    return StreamingResponse(
        service.stream_progress(task_id), media_type="text/event-stream"
    )


@router.post("/performance", response_model=PerformanceResponse)
@inject
async def check_engine_performance(
    request: PerformanceRequest,
    service: Annotated[
        CheckModelPerformanceService,
        Depends(Provide[RootContainer.evaluation.performance_service]),
    ],
):
    params = CheckModelPerformanceParams(
        dataset_name=request.dataset_name,
        engine=request.engine.model_dump(),
        n_samples=request.n_samples,
    )
    result = service.execute(params)

    return result.match(
        on_success=lambda value: PerformanceResponse(
            dataset_name=value["dataset_name"],
            solver_type=value["solver_type"],
            version=value["version"],
            insights=value["insights"],
        ),
        on_failure=lambda error: HTTPException(
            status_code=status.HTTP_404_NOT_FOUND
            if error.code == "NOT_FOUND"
            else status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": error.message,
                "error_code": error.code,
                "details": error.details,
            },
        ),
    )
