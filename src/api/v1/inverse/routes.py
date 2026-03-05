from fastapi import APIRouter, Depends, HTTPException

from ....modules.inverse.application.generate_candidates import (
    GenerateCandidatesService,
    GenerationConfig,
)
from ....modules.inverse.application.list_engines import ListEnginesService
from ....modules.inverse.application.train_inverse_mapping_engine import (
    SolverConfig,
    TrainInverseMappingEngineParams,
    TrainInverseMappingEngineService,
)
from .dependencies import (
    get_generation_service,
    get_list_engines_service,
    get_train_service,
)
from .schemas import (
    EngineListItem,
    GenerateRequest,
    GenerateResponse,
    TrainEngineRequest,
    TrainEngineResponse,
)

router = APIRouter()


@router.post("/train", response_model=TrainEngineResponse, status_code=201)
async def train_engine(
    request: TrainEngineRequest,
    service: TrainInverseMappingEngineService = Depends(get_train_service),
):
    """
    Train an inverse mapping engine for a dataset.
    """
    # Validate solver
    supported_solvers = ["GBPI", "MDN"]
    if request.solver.type not in supported_solvers:
        if request.solver.type in ["CVAE", "INN"]:
            raise HTTPException(
                status_code=422,
                detail=f"Solver type '{request.solver.type}' is not yet implemented. Coming soon.",
            )
        raise HTTPException(
            status_code=422,
            detail=f"Unknown solver type: '{request.solver.type}'. Supported: {supported_solvers}",
        )

    params = TrainInverseMappingEngineParams(
        dataset_name=request.dataset_name,
        solver=SolverConfig(type=request.solver.type, params=request.solver.params),
        transforms=request.transforms,
        split_ratio=request.split_ratio,
        random_state=request.random_state,
    )

    try:
        # Note: US1/T016 will update execute() to return a rich result dict.
        # For now, we handle identifying if it returns an entity or a dict.
        result = service.execute(params)

        if not isinstance(result, dict):
            # Temporary manual mapping until T016 is implemented
            return TrainEngineResponse(
                dataset_name=result.dataset_name,
                solver_type=result.solver.type(),
                engine_version=1,  # Placeholder
                status="completed",
                duration_seconds=0.0,
                n_train_samples=len(result.data_split.train_indices),
                n_test_samples=len(result.data_split.test_indices),
                split_ratio=result.data_split.split_ratio,
                loss_history=[],
                transform_summary=[],
            )
        return TrainEngineResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/generate", response_model=GenerateResponse)
async def generate_candidates(
    request: GenerateRequest,
    service: GenerateCandidatesService = Depends(get_generation_service),
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
        trust_radius=request.trust_radius,
        concentration_factor=request.concentration_factor,
        error_threshold=request.error_threshold,
    )

    try:
        result = service.execute(config)
        return GenerateResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.get("/engines/{dataset_name}", response_model=list[EngineListItem])
async def list_engines_for_dataset(
    dataset_name: str,
    service: ListEnginesService = Depends(get_list_engines_service),
):
    """
    List all trained engines for a specific dataset.
    """
    engines = service.execute(dataset_name)
    return [
        EngineListItem(
            solver_type=e["solver_type"],
            version=e["version"],
            created_at=e["created_at"],
        )
        for e in engines
    ]
