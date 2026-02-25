from pydantic import BaseModel, ConfigDict

from ..interfaces.base_estimator import BaseEstimator
from .estimator_params import (
    EstimatorParamsBase,
)


class TrainingLog(BaseModel):
    epochs: list[int] = []
    train_loss: list[float] = []
    val_loss: list[float] = []
    extras: dict[str, list[float]] = {}


class EstimatorStep(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: EstimatorParamsBase
    fitted: BaseEstimator
    training_log: TrainingLog
