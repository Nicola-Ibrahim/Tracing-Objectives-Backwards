from enum import StrEnum


class InverseSolverRegistry(StrEnum):
    GBPI = "GBPI"
    TDA_GBPI = "TDA-GBPI"
    HYBRID_GBPI = "HYBRID-GBPI"
    MDN = "MDN"
    INN = "INN"
