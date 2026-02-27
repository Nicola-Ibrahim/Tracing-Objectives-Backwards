from fastapi import APIRouter, Depends, HTTPException

from ...modules.generation.application.generate_candidates import (
    GenerateCoherentCandidatesService,
    GenerationConfig,
)
from ...modules.generation.application.prepare_context import (
    PrepareContextParams,
    PrepareContextService,
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
    )

    try:
        result = service.execute(config)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=500, detail="Internal server error during generation."
        )

    return GenerationResponse(
        pathway=result["pathway"],
        target_objective=result["target_objective"],
        candidate_decisions=result["candidate_decisions"],
        candidate_objectives=result["candidate_objectives"],
        residual_errors=result["residual_errors"],
        anchor_indices=result["anchor_indices"],
        is_inside_mesh=result["is_inside_mesh"],
        winner_index=result["winner_index"],
        winner_point=result["winner_point"],
        winner_decision=result["winner_decision"],
    )


@router.post("/train", response_model=TrainingResponse)
async def train_context(
    request: TrainingRequest,
    service: PrepareContextService = Depends(get_train_context_service),
):
    """
    Train (fit) surrogate models for a dataset context.
    """
    try:
        params = PrepareContextParams(dataset_name=request.dataset_name)
        service.execute(params)
        return TrainingResponse(dataset_name=request.dataset_name, status="completed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
