from enum import Enum


class InverseDecisionMapperType(Enum):
    """
    Defines the available types of interpolators that can be trained.
    This Enum serves as a central registry of interpolator kinds within the domain.
    """

    # Bivariate decision mappers
    # These are used for two-dimensional data interpolation
    NEURAL_NETWORK = "neural_network"
    GEODESIC = "geodesic"
    PCHIP = "pchip"
    SPLINE = "spline"
    POLYNOMIAL = "polynomial"
    RBF = "rbf"
    QUADRATIC = "quadratic"
    CUBIC = "cubic"
    NEAREST_NEIGHBORS = "nearest_neighbors"

    # Multivariate decision mappers
    # These are used for higher-dimensional data interpolation
    LINEAR_ND = "linear_nd"
    QUADRATIC_ND = "quadratic_nd"
