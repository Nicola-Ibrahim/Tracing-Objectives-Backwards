import numpy as np


class ObjectiveOutOfBoundsError(Exception):
    """
    Raised when a target objective is too far from the Pareto front.
    """

    def __init__(self, distance: float, suggestions: np.ndarray):
        self.distance = distance
        self.suggestions = suggestions
        self.message = (
            f"Objective is out of bounds (min dist = {distance:.4f}). "
            f"Suggested nearby objectives: {np.round(suggestions, 4).tolist()}"
        )
        super().__init__(self.message)
