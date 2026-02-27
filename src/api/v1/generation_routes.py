from fastapi import APIRouter, HTTPException

from ...modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...modules.generation.application.generate_candidates import (
    GenerateCoherentCandidatesService,
    GenerationConfig,
)
from ...modules.generation.infrastructure.repositories.context_repo import (
    FileSystemContextRepository,
)
from ...modules.shared.infrastructure.loggers.cmd_logger import CMDLogger
from ..schemas.generation import GenerationRequest, GenerationResponse

router = APIRouter()

# Instantiate dependencies
context_repo = FileSystemContextRepository()
dataset_repo = FileSystemDatasetRepository()
logger = CMDLogger(name="GenerationAPI")
service = GenerateCoherentCandidatesService(
    context_repository=context_repo,
    dataset_repository=dataset_repo,
    logger=logger,
)


@router.post("/generate", response_model=GenerationResponse)
async def generate_candidates(request: GenerationRequest):
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
        logger.log_error(f"Generation failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Internal server error during generation."
        )

    return GenerationResponse(
        pathway=result.pathway,
        target_objective=tuple(result.target_objective.flatten().tolist()),
        candidate_decisions=[row.tolist() for row in result.candidates],
        candidate_objectives=[
            tuple(row) for row in result.predicted_objectives.tolist()
        ],
        residual_errors=result.residual_errors.tolist(),
        anchor_indices=result.anchor_indices,
        is_inside_mesh=result.is_inside_mesh,
    )
