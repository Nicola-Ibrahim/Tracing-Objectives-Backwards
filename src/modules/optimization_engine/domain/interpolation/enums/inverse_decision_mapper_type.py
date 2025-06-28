from enum import Enum


class InverseDecisionMapperType(Enum):
    """
    Defines the available types of interpolators that can be trained.
    This Enum serves as a central registry of interpolator kinds within the domain.
    """

    # Multivariate decision mappers
    LINEAR_ND = "linear_nd"
    NEAREST_NEIGHBORS_ND = "nearest_neighbors_nd"
    RBF_ND = "rbf_nd"
    CLOUGH_TOCHER_ND = "clough_tocher_nd"
    NEURAL_NETWORK_ND = "neural_network_nd"
