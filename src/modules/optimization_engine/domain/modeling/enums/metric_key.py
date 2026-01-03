from enum import Enum


class DefaultValidationMetricEnum(str, Enum):
    MSE = "MSE"
    MAE = "MAE"
    R2 = "R2"
