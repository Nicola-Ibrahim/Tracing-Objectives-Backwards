from enum import Enum


class InverseSolverRegistry(str, Enum):
    GBPI = "GBPI"
    TDA_GBPI = "TDA-GBPI"
    PROBABILISTIC = "probabilistic"
    MDN = "MDN"  # Added as it was present in current factory logic
