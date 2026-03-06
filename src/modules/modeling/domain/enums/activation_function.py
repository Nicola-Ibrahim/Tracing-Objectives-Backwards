from enum import Enum


class ActivationFunctionEnum(Enum):
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SOFTPLUS = "softplus"
    IDENTITY = "identity"
    ELU = "elu"
