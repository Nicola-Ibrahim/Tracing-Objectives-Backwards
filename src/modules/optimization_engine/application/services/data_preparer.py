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
        X_normalizer: BaseNormalizer,
        y_normalizer: BaseNormalizer,
        test_size: float = 0.2,
        random_state: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of rows")

        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        X_train = X_normalizer.fit_transform(X_train_raw)
        X_test = X_normalizer.transform(X_test_raw)
        y_train = y_normalizer.fit_transform(y_train_raw)
        y_test = y_normalizer.transform(y_test_raw)

        return X_train, X_test, y_train, y_test
