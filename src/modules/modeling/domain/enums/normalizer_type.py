from enum import Enum


class NormalizerTypeEnum(str, Enum):
    MIN_MAX = "min_max"
    STANDARD = "standard"
    UNIT_VECTOR = "unit_vector"
    LOG = "log"
    HYPERCUBE = "hypercube"
