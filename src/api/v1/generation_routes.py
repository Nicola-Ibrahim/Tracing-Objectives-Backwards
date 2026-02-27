from fastapi import APIRouter, Depends, HTTPException

from ...modules.generation.application.generate_candidates import (
    GenerateCoherentCandidatesService,
    GenerationConfig,
)
from ..dependencies import get_generation_service
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
        service.logger.log_error(f"Generation failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Internal server error during generation."
        )

    return GenerationResponse(
        pathway=result["pathway"],
        target_objective=result["target_objective"],
        candidate_decisions=[row.tolist() for row in result["candidate_decisions"]],
        candidate_objectives=[
            tuple(row) for row in result["candidate_objectives"].tolist()
        ],
        residual_errors=result["residual_errors"].tolist(),
        anchor_indices=result["anchor_indices"],
        is_inside_mesh=result["is_inside_mesh"],
    )
