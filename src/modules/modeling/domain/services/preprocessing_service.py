import numpy as np
from sklearn.model_selection import train_test_split

from ..interfaces.base_transform import BaseTransformer
from ..value_objects.split_step import SplitConfig, SplitStep


class PreprocessingService:
    """
    Domain service for handling data splitting and pipeline transformations.
    """

    def split(
        self, X: np.ndarray, y: np.ndarray, config: SplitConfig
    ) -> tuple[SplitStep, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits data according to configuration.
        Returns the SplitStep (metadata) and the split arrays: X_train, X_test, y_train, y_test.
        """
        if config.strategy == "holdout":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config.test_size, random_state=config.random_state
            )
        else:
            raise NotImplementedError(
                f"Split strategy {config.strategy} not supported yet."
            )

        step = SplitStep(config=config)
        return step, X_train, X_test, y_train, y_test

    def apply_transforms(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        transforms: list[BaseTransformer],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fits transforms sequentially on training data and applies them to both train and test data.
        """
        for transform in transforms:
            transform.fit(X_train)
            X_train = transform.transform(X_train)
            if X_test is not None:
                X_test = transform.transform(X_test)

        return X_train, X_test
