from enum import StrEnum


class ScaleMethod(StrEnum):
    SD = "sd"
    MINMAX = "minmax"
    ROBUST = "robust"
