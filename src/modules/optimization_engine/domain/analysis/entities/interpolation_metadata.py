class ModelMetrics(BaseModel):
    """Represents the metrics dictionary."""

    MeanSquaredErrorValidationMetric: float
    model_config = model_config_strict


class ModelMetadata(BaseModel):
    """The main Pydantic model for the metadata.json file."""

    id: str
    name: str
    parameters: Union[
        GaussianProcessEstimatorParams,
        RBFEstimatorParams,
        NearestNeighborsEstimatorParams,
    ] = Field(
        ...,
        discriminator="type",  # Use a discriminator to tell Pydantic which sub-model to use based on the 'type' field
        description="Parameters specific to the mapper type.",
    )
    metrics: ModelMetrics
    trained_at: datetime
    model_config = model_config_strict
