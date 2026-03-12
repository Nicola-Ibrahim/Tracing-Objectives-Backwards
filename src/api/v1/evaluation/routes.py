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
    DomainAssessmentData,
    MetricSeries,
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
            capabilities={
                f"{report.engine.type} (v{report.engine.version})": report.engine.capability.value
                for report in reports
            },
            objective_space=DomainAssessmentData(
                ecdf={
                    f"{report.engine.type} (v{report.engine.version})": MetricSeries(
                        x=report.objective_space.ecdf_profile.x_values,
                        y=report.objective_space.ecdf_profile.cumulative_probabilities,
                    )
                    for report in reports
                },
                metrics={
                    f"{report.engine.type} (v{report.engine.version})": {
                        "mean_best_shot": report.objective_space.mean_best_shot,
                        "median_best_shot": report.objective_space.median_best_shot,
                        "mean_bias": report.objective_space.mean_bias,
                        "mean_dispersion": report.objective_space.mean_dispersion,
                    }
                    for report in reports
                },
            ),
            decision_space=DomainAssessmentData(
                ecdf={
                    f"{report.engine.type} (v{report.engine.version})": MetricSeries(
                        x=report.decision_space.ecdf_profile.x_values,
                        y=report.decision_space.ecdf_profile.cumulative_probabilities,
                    )
                    for report in reports
                    if hasattr(report.decision_space, "ecdf_profile")
                },
                calibration_curves={
                    f"{report.engine.type} (v{report.engine.version})": MetricSeries(
                        x=report.decision_space.calibration_curve.nominal_coverage,
                        y=report.decision_space.calibration_curve.empirical_coverage,
                    )
                    for report in reports
                    if hasattr(report.decision_space, "calibration_curve")
                },
                metrics={
                    f"{report.engine.type} (v{report.engine.version})": (
                        {
                            "mace": report.decision_space.mace,
                            "mean_crps": report.decision_space.mean_crps,
                            "interval_width": report.decision_space.mean_interval_width,
                            "diversity": report.decision_space.mean_diversity,
                        }
                        if hasattr(report.decision_space, "pit_profile")
                        else {
                            "mace": report.decision_space.mean_coverage_error,
                            "interval_width": report.decision_space.mean_interval_width,
                            "winkler_score": report.decision_space.mean_winkler_score,
                        }
                    )
                    for report in reports
                },
            ),
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
