from enum import Enum


class InverseSolverRegistry(str, Enum):
    GBPI = "GBPI"
    TDA_GBPI = "TDA-GBPI"
    HYBRID_GBPI = "HYBRID-GBPI"
    MDN = "MDN"
