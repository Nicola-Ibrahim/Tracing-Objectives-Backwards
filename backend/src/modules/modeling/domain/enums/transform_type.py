from enum import StrEnum


class TransformTypeEnum(StrEnum):
    MIN_MAX = "min_max"
    STANDARD = "standard"
    UNIT_VECTOR = "unit_vector"
    LOG = "log"
    HYPERCUBE = "hypercube"
