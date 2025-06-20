from ....interpolation.application import train_interpolator_handler
from ...infrastructure.inverse_decision_mappers.splines.linear import LinearInterpolator
from ...infrastructure.loggers.wandb_logger import WandbLogger
from ...infrastructure.metrics import mean_squared_error
from ...infrastructure.similarities import cosine_similarity

# --- IMPORTANT: You will need to import your concrete interpolator implementations here ---
# Example:
# from ..interpolators.neural_network_interpolator import NNInterpolator
# from ..interpolators.geodesic_interpolator import GeodesicInterpolator
# from ..interpolators.linear_interpolator import LinearInterpolator # If you have one


# Initialize WandB logger
logger = WandbLogger(
    project="Interpolation-models",
)

interpolator = LinearInterpolator(similarity_metric=cosine_similarity)

train_interpolator_handler(
    interpolator=interpolator,
    interpolator_name="F1",
    interpolator_params={},
    logger=logger,
    validation_metric=mean_squared_error,
)
