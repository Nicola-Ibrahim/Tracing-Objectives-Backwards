from enum import Enum


class InterpolatorType(Enum):
    """
    Defines the available types of interpolators that can be trained.
    This Enum serves as a central registry of interpolator kinds within the domain.
    """

    NEURAL_NETWORK = "neural_network"
    GEODESIC = "geodesic"
    LINEAR = "linear"
    K_NEAREST_NEIGHBOR = "k_nearest_neighbor"
