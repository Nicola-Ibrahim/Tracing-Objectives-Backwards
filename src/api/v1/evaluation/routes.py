from fastapi import APIRouter, Depends, HTTPException, status

from ....modules.evaluation.application.check_engine_performance import (
    CheckModelPerformanceParams,
    CheckModelPerformanceService,
)
from ....modules.evaluation.application.diagnose_engines import (
    RunDiagnosticsCommand,
    RunDiagnosticsService,
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
    service: RunDiagnosticsService = Depends(get_diagnose_service),
):
    params = RunDiagnosticsCommand(
        dataset_name=request.dataset_name,
        inverse_engine_candidates=[c.model_dump() for c in request.candidates],
        num_samples=request.num_samples,
        scale_method=request.scale_method,
    )

    result = service.execute(params)
    return result.match(
        on_success=lambda reports: DiagnoseResponse(
            dataset_name=request.dataset_name,
            engines=[
                f"{report.engine.type} (v{report.engine.version})" for report in reports
            ],
            ecdf={
                f"{report.engine.type} (v{report.engine.version})": {
                    "x": report.objective_space.ecdf_profile.x_values,
                    "y": report.objective_space.ecdf_profile.cumulative_probabilities,
                }
                for report in reports
            },
            pit={
                f"{report.engine.type} (v{report.engine.version})": (
                    {
                        "x": report.decision_space.calibration_curve.nominal_coverage,
                        "y": report.decision_space.calibration_curve.empirical_coverage,
                    }
                    if hasattr(report.decision_space, "calibration_curve")
                    else {
                        "x": report.decision_space.ecdf_profile.x_values,
                        "y": report.decision_space.ecdf_profile.cumulative_probabilities,
                    }
                )
                for report in reports
            },
            mace={
                f"{report.engine.type} (v{report.engine.version})": (
                    report.decision_space.mace
                    if hasattr(report.decision_space, "mace")
                    else report.decision_space.mean_coverage_error
                )
                for report in reports
            },
            warnings=[],
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


# @router.post("/diagnose/cached", response_model=DiagnoseResponse)
# async def get_cached_diagnose(
#     request: DiagnoseRequest,
#     service: DiagnoseInverseModelsService = Depends(get_diagnose_service),
# ):
#     """
#     Retrieve existing diagnostics from database if available.
#     Returns 404 if any engine in the request has no cached diagnostics.
#     """
#     candidates = [
#         InverseEngineCandidate(solver_type=c.solver_type, version=c.version)
#         for c in request.candidates
#     ]
#     params = DiagnoseInverseModelsParams(
#         dataset_name=request.dataset_name,
#         inverse_engine_candidates=candidates,
#         num_samples=request.num_samples,
#         scale_method=request.scale_method,
#     )

#     result = service.get_cached_diagnostics(params)
#     return result.match(
#         on_success=lambda diagnostics: DiagnoseResponse(
#             dataset_name=diagnostics.dataset_name,
#             engine_diagnostics=diagnostics.engine_diagnostics,
#         ),
#         on_failure=lambda error: HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail={
#                 "message": error.message,
#                 "error_code": error.code,
#                 "details": error.details,
#             },
#         ),
#     )


@router.post("/performance", response_model=PerformanceResponse)
async def check_engine_performance(
    request: PerformanceRequest,
    service: CheckModelPerformanceService = Depends(get_performance_service),
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
