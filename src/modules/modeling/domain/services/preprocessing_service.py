
import numpy as np
from sklearn.model_selection import train_test_split

from ..interfaces.base_transform import BaseTransformStep
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
        y_train: np.ndarray,
        y_test: np.ndarray,
        transforms: list[BaseTransformStep],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fits transforms on training data and applies them to both train and test data.
        """
        from ..interfaces.base_transform import TransformTarget

        for transform in transforms:
            if transform.target in (TransformTarget.DECISIONS, TransformTarget.BOTH):
                transform.fit(X_train)
                X_train = transform.transform(X_train)
                X_test = transform.transform(X_test)

            if transform.target in (TransformTarget.OBJECTIVES, TransformTarget.BOTH):
                # We need to distinguish if the transform is meant to be fit on y or X, but the interface just says fit(X).
                # Wait: if target is DECISIONS, it fits/transforms X. If OBJECTIVES, it fits/transforms y.
                # Let's adjust for Target:
                if transform.target == TransformTarget.OBJECTIVES:
                    transform.fit(y_train)
                    y_train = transform.transform(y_train)
                    y_test = transform.transform(y_test)
                elif transform.target == TransformTarget.BOTH:
                    # In this setup, BOTH would mean we fit on concatenated or we only apply to decisions?
                    # The spec mostly deals with NormalizationStep mapping to decisions/objectives separately.
                    pass

        return X_train, X_test, y_train, y_test
