from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from ..interpolation.interfaces.base_inverse_decision_mappers import (
    BaseInverseDecisionMapper,
)
from ..interpolation.interfaces.base_normalizer import BaseNormalizer


class DecisionMapperTrainingService:
    """
    Domain service responsible for handling the data splitting, normalization,
    and the core training/prediction workflow for interpolators.
    It separates the training and prediction concerns for better SRP adherence.
    """

    def train(
        self,
        inverse_decision_mapper: BaseInverseDecisionMapper,
        X_data: NDArray[np.floating],
        Y_data: NDArray[np.floating],
        x_normalizer_class: type[BaseNormalizer],
        y_normalizer_class: type[BaseNormalizer],
        test_size: float = 0.33,
        random_state: int = 42,
    ) -> Tuple[
        BaseInverseDecisionMapper,
        NDArray[np.floating],
        NDArray[np.floating],
        BaseNormalizer,  # Return the fitted x_normalizer instance
        BaseNormalizer,  # Return the fitted y_normalizer instance
    ]:
        """
        Splits data, normalizes it, and trains the interpolator.

        Args:
            inverse_decision_mapper: An unfitted instance of BaseInverseDecisionMapper to be trained.
            X_data: Raw input data (candidate solutions).
            Y_data: Raw output data (objective front).
            x_normalizer_class: The class type for the X-data normalizer.
            y_normalizer_class: The class type for the Y-data normalizer.
            test_size: Proportion of data for validation.
            random_state: Seed for reproducibility.

        Returns:
            A tuple containing:
            - fitted_inverse_decision_mapper (BaseInverseDecisionMapper): The interpolator instance after fitting.
            - X_val_norm (NDArray): Normalized validation input features.
            - y_val (NDArray): True validation output values (original scale).
            - x_normalizer (BaseNormalizer): The fitted normalizer for X data.
            - y_normalizer (BaseNormalizer): The fitted normalizer for Y data.
        """
        # Split data into train and validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X_data, Y_data, test_size=test_size, random_state=random_state
        )

        # Instantiate normalizers
        x_normalizer = x_normalizer_class(feature_range=(-1, 1))  # Example range
        y_normalizer = y_normalizer_class(feature_range=(0, 1))  # Example range

        # Normalize training data
        X_train_norm = x_normalizer.fit_transform(X_train)
        Y_train_norm = y_normalizer.fit_transform(y_train)

        # Transform validation data using same parameters
        X_val_norm = x_normalizer.transform(X_val)

        # Fit the interpolator instance
        inverse_decision_mapper.fit(
            candidate_solutions=X_train_norm,
            objective_front=Y_train_norm,
        )

        # Return fitted instance, normalized validation X, true validation Y, and fitted normalizers
        return inverse_decision_mapper, X_val_norm, y_val, x_normalizer, y_normalizer

    def predict(
        self,
        fitted_inverse_decision_mapper: BaseInverseDecisionMapper,
        X_query_norm: NDArray[np.floating],
        y_normalizer_instance: BaseNormalizer,
    ) -> NDArray[np.floating]:
        """
        Generates predictions using a fitted interpolator and inverse-transforms them.

        Args:
            fitted_inverse_decision_mapper: A pre-fitted instance of BaseInverseDecisionMapper.
            X_query_norm: Normalized input features for which to make predictions.
            y_normalizer_instance: The *fitted* normalizer used for Y data during training.

        Returns:
            y_pred (NDArray): Predicted output values in their original scale.
        """
        # Predict normalized values
        y_pred_norm = fitted_inverse_decision_mapper.predict(X_query_norm)

        # Inverse-transform predictions to original scale
        y_pred = y_normalizer_instance.inverse_transform(y_pred_norm)

        return y_pred
