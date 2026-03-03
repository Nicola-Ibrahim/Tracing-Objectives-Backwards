from fastapi import APIRouter, Depends, HTTPException

from ...modules.generation.application.generate_candidates import (
    GenerateCoherentCandidatesService,
    GenerationConfig,
)
from ...modules.generation.application.train_context import (
    TrainContextParams,
    TrainContextService,
)
from ..dependencies import get_generation_service, get_train_context_service
from ..schemas.dataset import TrainingRequest, TrainingResponse
from ..schemas.generation import GenerationRequest, GenerationResponse

router = APIRouter()


@router.post("/generate", response_model=GenerationResponse)
async def generate_candidates(
    request: GenerationRequest,
    service: GenerateCoherentCandidatesService = Depends(get_generation_service),
):
    """
    Generate coherent candidates for a target objective.
    """
    config = GenerationConfig(
        dataset_name=request.dataset_name,
        target_objective=request.target_objective,
        n_samples=request.n_samples,
        trust_radius=request.trust_radius,
        concentration_factor=request.concentration_factor,
        error_threshold=request.error_threshold,
    )

    try:
        result = service.execute(config)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Generation error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error during generation: {str(e)}"
        )

    return GenerationResponse(
        pathway=result["pathway"],
        target_objective=result["target_objective"],
        candidate_decisions=result["candidate_decisions"],
        candidate_objectives=result["candidate_objectives"],
        objective_space_residual_sorted=result["objective_space_residual_sorted"],
        vertices_indices=result["vertices_indices"],
        is_simplex_found=result["is_simplex_found"],
        is_coherent=result["is_coherent"],
        best_index=result["best_index"],
        best_objective=result["best_objective"],
        best_decision=result["best_decision"],
        tau=result["tau"],
        vertice_distances=result["vertice_distances"],
        all_residuals=result["all_residuals"],
    )


@router.post("/train", response_model=TrainingResponse)
async def train_context(
    request: TrainingRequest,
    service: TrainContextService = Depends(get_train_context_service),
):
    """
    Train (fit) surrogate models for a dataset context.
    """
    try:
        params = TrainContextParams(dataset_name=request.dataset_name)
        service.execute(params)
        return TrainingResponse(dataset_name=request.dataset_name, status="completed")
    except Exception as e:
        print(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
