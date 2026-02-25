import click

from ...modules.generation.application.use_cases.generate_candidates import (
    GenerateCoherentCandidatesService,
)
from ...modules.generation.domain.value_objects.coherence_params import CoherenceParams
from ...modules.generation.infrastructure.repositories.context_repo import (
    FileSystemContextRepository,
)
from ...modules.modeling.infrastructure.repositories.trained_pipeline_repo import (
    FileSystemTrainedPipelineRepository,
)
from ...modules.shared.infrastructure.loggers.cmd_logger import CMDLogger


@click.command()
@click.option("--dataset-name", required=True, type=str, help="Name of the dataset")
@click.option(
    "--target",
    required=True,
    nargs=2,
    type=float,
    help="Target objective coordinates (2D)",
)
@click.option("--n-samples", type=int, default=50, help="Number of sampled candidates")
@click.option(
    "--concentration-factor",
    type=float,
    default=10.0,
    help="Concentration factor for Dirichlet sampling",
)
@click.option(
    "--trust-radius",
    type=float,
    default=0.05,
    help="Trust region radius for optimization fallback",
)
@click.option(
    "--error-threshold",
    type=float,
    default=None,
    help="Threshold to filter candidates with high predicted residual error",
)
def main(
    dataset_name: str,
    target: tuple[float, float],
    n_samples: int,
    concentration_factor: float,
    trust_radius: float,
    error_threshold: float | None,
):
    """
    Real-time step: Generate physically coherent design candidates for a specific target objective.
    """
    logger = CMDLogger(name="GenerateCoherentCandidatesLogger")
    service = GenerateCoherentCandidatesService(
        context_repository=FileSystemContextRepository(),
        model_repository=FileSystemTrainedPipelineRepository(),
        logger=logger,
    )

    params = CoherenceParams(
        n_samples=n_samples,
        concentration_factor=concentration_factor,
        trust_radius=trust_radius,
        error_threshold=error_threshold,
        k_neighbors=5,
        tau_percentile=95.0,
    )

    result = service.execute(dataset_name, target, params)

    click.echo(f"Generation successful. Pathway: {result.pathway}")
    click.echo(f"Valid candidates: {len(result.candidates)}")
    for i in range(min(5, len(result.candidates))):
        click.echo(
            f"  [{i}] Error: {result.residual_errors[i]:.4f} | Pred: {result.predicted_objectives[i].tolist()} | X: {result.candidates[i].tolist()[:5]}..."
        )


if __name__ == "__main__":
    main()
