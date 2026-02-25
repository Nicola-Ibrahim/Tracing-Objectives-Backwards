from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class CoherenceParams(BaseModel):
    """
    User-configurable parameters for the generation pipeline.
    """

    model_config = ConfigDict(frozen=True)

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
    error_threshold: Optional[float] = Field(
        default=None,
        description="Residual error cutoff for filtering; None = no filtering",
    )
    k_neighbors: int = Field(
        default=5, ge=1, description="k for nearest-neighbor tau computation"
    )
    tau_percentile: float = Field(
        default=95.0, gt=0, le=100, description="Percentile for coherence threshold"
    )

    def model_post_init(self, __context):
        if self.error_threshold is not None and self.error_threshold <= 0:
            raise ValueError(
                "error_threshold must be strictly greater than 0 if provided"
            )
