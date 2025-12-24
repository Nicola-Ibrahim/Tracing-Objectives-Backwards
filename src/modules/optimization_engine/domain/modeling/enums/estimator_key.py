from enum import Enum


class EstimatorKeyEnum(str, Enum):
    COCO = "coco"
    CVAE = "cvae"
    GAUSSIAN_PROCESS = "gaussian_process"
    MDN = "mdn"
    NEURAL_NETWORK = "neural_network"
    RBF = "rbf"
