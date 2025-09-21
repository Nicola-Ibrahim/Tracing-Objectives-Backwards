from enum import Enum


class DiversityMethod(str, Enum):
    """Available diversity methods for generating suggestions."""

    MAX_MIN_DISTANCE = "max_min_distance"
    KMEANS = "kmeans"
    EUCLIDEAN = "euclidean"
