from enum import StrEnum


class ActivationFunctionEnum(StrEnum):
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SOFTPLUS = "softplus"
    IDENTITY = "identity"
    ELU = "elu"
