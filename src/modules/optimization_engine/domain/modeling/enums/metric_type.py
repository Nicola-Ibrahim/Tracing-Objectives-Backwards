from enum import Enum


class MetricTypeEnum(str, Enum):
    MSE = "MSE"
    MAE = "MAE"
    R2 = "R2"
    NEGATIVE_LOG_LIKELIHOOD = "Negative Log-Likelihood"
    # Alias (optional): allows using 'NLL' while keeping a single canonical value
    NLL = "Negative Log-Likelihood"
