from enum import Enum


class OptimizerFunctionEnum(Enum):
    SGD = "sgd"
    ADAM = "adam"
    RMSPROP = "rmsprop"
    GRADIENT_DESCENT = "gradient_descent"
