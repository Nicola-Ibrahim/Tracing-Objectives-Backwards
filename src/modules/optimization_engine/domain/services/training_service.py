from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from ..interpolation.entities.interpolator_model import InterpolatorModel
from ..interpolation.interfaces.base_interpolator import BaseInterpolator
from ..interpolation.interfaces.base_normalizer import BaseNormalizer


class TrainingService:
    """
    Domain service responsible for handling the data splitting, normalization,
    and the core training/prediction workflow for interpolators.
    """

    def train_and_predict(
        self,
        instance: BaseInterpolator,
        name: str,
        type: str,
        params: dict,
        X_data: NDArray[np.floating],
        Y_data: NDArray[np.floating],
        x_normalizer_class: type[BaseNormalizer],
        y_normalizer_class: type[BaseNormalizer],
        test_size: float = 0.33,
        random_state: int = 42,
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating], InterpolatorModel]:
        """
        Splits data, normalizes it, trains the interpolator, makes predictions,
        and returns the validation results along with the trained model entity.
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
        instance.fit(
            candidate_solutions=X_train_norm,
            objective_front=Y_train_norm,
        )

        # Predict normalized values on validation set
        y_pred_norm = instance.generate(X_val_norm)

        # Inverse-transform predictions to original scale
        y_pred = y_normalizer.inverse_transform(y_pred_norm)

        # Create the InterpolatorModel entity
        interpolator_model = InterpolatorModel(
            name=name,
            type=type.value,  # Use .value for enum string
            parameters=params,
            fitted_interpolator=instance,
            description=f"{name} trained via {type.name} method.",
            # Metrics will be added by the command handler
        )

        # Return y_val (true values) and y_pred (predicted values) for external metric calculation,
        # along with the newly created InterpolatorModel entity.
        return y_val, y_pred, interpolator_model
