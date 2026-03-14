from enum import StrEnum


class OptimizerFunctionEnum(StrEnum):
    SGD = "sgd"
    ADAM = "adam"
    RMSPROP = "rmsprop"
    GRADIENT_DESCENT = "gradient_descent"
