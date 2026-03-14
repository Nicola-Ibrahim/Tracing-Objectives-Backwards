from enum import StrEnum


class EngineCapability(StrEnum):
    FULL_DISTRIBUTION = "full_distribution"  # MDN, CAVA
    PREDICTION_INTERVAL = "prediction_interval"  # GBPI
