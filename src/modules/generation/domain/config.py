from pydantic import BaseModel, ConfigDict, Field


class GenerationConfig(BaseModel):
    """
    User-configurable parameters for the generation pipeline.
    Combines dynamic execution parameters and coherence tolerances.
    """

    model_config = ConfigDict(frozen=True)

    dataset_name: str = Field(..., description="Name of the dataset")
    target_objective: tuple[float, float] = Field(
        ..., description="Target objective coordinates (2D)"
    )
    n_samples: int = Field(
        default=50, ge=1, description="Number of Dirichlet weight samples (Phase 3A)"
    )
    concentration_factor: float = Field(
        default=10.0,
        gt=0,
        description="Controls tightness of Dirichlet sampling around target",
    )
    trust_radius: float = Field(
        default=0.05,
        gt=0,
        le=1,
        description="Trust-region radius in normalized decision space (Phase 3B)",
    )
    error_threshold: float | None = Field(
        default=None,
        description="Residual error cutoff for filtering; None = no filtering",
    )

    def model_post_init(self, __context):
        if self.error_threshold is not None and self.error_threshold <= 0:
            raise ValueError(
                "error_threshold must be strictly greater than 0 if provided"
            )
