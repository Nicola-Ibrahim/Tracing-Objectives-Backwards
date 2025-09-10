from enum import Enum


class EstimatorTypeEnum(Enum):
    """
    Defines the available types of interpolators that can be trained.
    This Enum serves as a central registry of interpolator kinds within the domain.
    """

    CLOUGH_TOCHER_ND = "clough_tocher_nd"
    NEURAL_NETWORK_ND = "neural_network_nd"
    NEAREST_NEIGHBORS_ND = "nearest_neighbors_nd"
    LINEAR_ND = "linear_nd"
    RBF = "RBF"
    GAUSSIAN_PROCESS_ND = "gaussian_process_nd"
    GEODESIC_ND = "geodesic_nd"
    SPLINE_ND = "spline_nd"
    KRIGING_ND = "kriging_nd"
    SVR_ND = "svr_nd"
    CVAE = "CVAE"
    MDN = "MDN"
