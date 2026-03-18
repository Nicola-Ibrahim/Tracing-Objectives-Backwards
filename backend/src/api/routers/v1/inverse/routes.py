from typing import Annotated

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException, status

from src.modules.inverse.application.inverse_service import (
    GenerationConfig,
    InverseService,
    SolverConfig,
    TrainInverseMappingEngineParams,
)
from src.modules.inverse.infrastructure.config.di import InverseContainer

from .schemas import (
    EngineListItem,
    GenerateRequest,
    GenerateResponse,
    SolversDiscoveryResponse,
    TrainEngineRequest,
    TrainEngineResponse,
)

router = APIRouter(prefix="/inverse", tags=["Inverse Mapping"])


@router.get("/solvers", response_model=SolversDiscoveryResponse)
@inject
async def list_available_solvers(
    service: Annotated[
        InverseService, Depends(Provide[InverseContainer.inverse_service])
    ],
):
    """
    List all available inverse mapping solvers and their required parameters.
    """
    result = service.get_available_solvers()
    return result.match(
        on_success=lambda schemas: SolversDiscoveryResponse(solvers=schemas),
        on_failure=lambda error: HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": error.message,
                "error_code": error.code,
                "details": error.details,
            },
        ),
    )


@router.post("/train", response_model=TrainEngineResponse, status_code=201)
@inject
async def train_engine(
    request: TrainEngineRequest,
    service: Annotated[
        InverseService, Depends(Provide[InverseContainer.inverse_service])
    ],
):
    """
    Train an inverse mapping engine for a dataset.
    """
    params = TrainInverseMappingEngineParams(
        dataset_name=request.dataset_name,
        solver=SolverConfig(type=request.solver.type, params=request.solver.params),
        transforms=request.transforms,
    )
    result = service.train_engine(params)
    return result.match(
        on_success=lambda value: TrainEngineResponse(
            dataset_name=value["dataset_name"],
            solver_type=value["solver_type"],
            engine_version=value["engine_version"],
            status=value["status"],
            duration_seconds=value["duration_seconds"],
            n_train_samples=value["n_train_samples"],
            n_test_samples=value["n_test_samples"],
            split_ratio=value["split_ratio"],
            training_history=value["training_history"],
            transform_summary=value["transform_summary"],
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


@router.post("/generate", response_model=GenerateResponse)
@inject
async def generate_candidates(
    request: GenerateRequest,
    service: Annotated[
        InverseService, Depends(Provide[InverseContainer.inverse_service])
    ],
):
    """
    Generate candidate designs for a target objective using a trained engine.
    """
    config = GenerationConfig(
        dataset_name=request.dataset_name,
        target_objective=request.target_objective,
        solver_type=request.solver_type,
        version=request.version,
        n_samples=request.n_samples,
    )

    result = service.generate_candidates(config)
    return result.match(
        on_success=lambda value: GenerateResponse(
            solver_type=value["solver_type"],
            target_objective=value["target_objective"],
            candidate_decisions=value["candidate_decisions"],
            candidate_objectives=value["candidate_objectives"],
            best_index=value["best_index"],
            best_candidate_objective=value["best_candidate_objective"],
            best_candidate_decision=value["best_candidate_decision"],
            best_candidate_residual=value["best_candidate_residual"],
            metadata=value["metadata"],
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


@router.get("/engines", response_model=list[EngineListItem])
@inject
async def list_all_engines(
    service: Annotated[
        InverseService, Depends(Provide[InverseContainer.inverse_service])
    ],
):
    """
    List all trained engines across all datasets (Inference Hub).
    """
    result = service.list_engines(None)
    return result.match(
        on_success=lambda value: [
            EngineListItem(
                dataset_name=item.get("dataset_name"),
                solver_type=item["solver_type"],
                version=item["version"],
                created_at=item["created_at"],
            )
            for item in value
        ],
        on_failure=lambda error: HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": error.message,
                "error_code": error.code,
                "details": error.details,
            },
        ),
    )


@router.get(
    "/engines/{dataset_name}/{solver_type}/{version}",
    # response_model=EngineDetailResponse,
)
@inject
async def get_engine_details(
    dataset_name: str,
    solver_type: str,
    version: int,
    service: Annotated[
        InverseService, Depends(Provide[InverseContainer.inverse_service])
    ],
):
    """
    Retrieve full details for a specific engine version.
    """
    result = service.get_engine_details(dataset_name, solver_type, version)
    if result.is_failure:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND
            if result.error.code == "NOT_FOUND"
            else status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": result.error.message,
                "error_code": result.error.code,
                "details": result.error.details,
            },
        )
    return result.value
