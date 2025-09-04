import numpy as np
from sklearn.model_selection import train_test_split

from ...domain.model_management.interfaces.base_normalizer import BaseNormalizer


class DataPreparer:
    """
    Responsible for splitting and normalising datasets.
    Stateless and deterministic.
    """

    @staticmethod
    def single_split(
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of rows")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        return X_train, X_test, y_train, y_test
