from ..infrastructure.interpolators.linear import LinearInterpolator
from ..infrastructure.loggers.wandb_logger import WandbLogger
from ..application.train_model import train_model
from ..domain.services.metrics import mean_squared_error
from ..domain.services.similarities import cosine_similarity

# Initialize WandB logger
logger = WandbLogger(
    project="Interpolation-models",
)

interpolator = LinearInterpolator(similarity_metric=cosine_similarity)

train_model(
    interpolator=interpolator,
    interpolator_name="F1",
    interpolator_params={},
    logger=logger,
    validation_metric=mean_squared_error,
)
