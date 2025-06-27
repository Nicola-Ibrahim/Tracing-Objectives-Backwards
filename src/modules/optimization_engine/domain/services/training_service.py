import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from ..interpolation.interfaces.base_inverse_decision_mappers import (
    BaseInverseDecisionMapper,
)
from ..interpolation.interfaces.base_metric import BaseValidationMetric
from ..interpolation.interfaces.base_normalizer import BaseNormalizer


class DecisionMapperTrainingService:
    """
    Domain service responsible for handling the data splitting, normalization,
    and the core training/prediction workflow for interpolators.
    It encapsulates the training and prediction concerns for better SRP adherence.
    """

    def __init__(
        self,
        validation_metric: BaseValidationMetric,
        decisions_normalizer: BaseNormalizer,
        objectives_normalizer: BaseNormalizer,
    ):
        """
        Initializes the service with a validation metric dependency.
        """
        self._validation_metric = validation_metric
        self._decisions_normalizer = decisions_normalizer
        self._objectives_normalizer = objectives_normalizer

    def train(
        self,
        inverse_decision_mapper: BaseInverseDecisionMapper,
        objectives: NDArray[np.floating],  # Objective data (input to the mapper)
        decisions: NDArray[np.floating],  # Decision data (output from the mapper)
        test_size: float = 0.33,
        random_state: int = 42,
    ) -> tuple[
        BaseInverseDecisionMapper,
        dict[str, float],
    ]:
        """
        Splits data, normalizes it, trains the interpolator, and calculates validation metrics.

        Args:
            inverse_decision_mapper: An unfitted instance of BaseInverseDecisionMapper.
            objectives: Raw input data for the mapper (Objectives).
            decisions: Raw output data for the mapper (Decisions).
            test_size: Proportion of data for validation.
            random_state: Seed for reproducibility.

        Returns:
            A tuple containing:
            - fitted_inverse_decision_mapper (BaseInverseDecisionMapper): The trained interpolator.
            - objectives_normalizer (BaseNormalizer): The fitted normalizer for objective data.
            - decisions_normalizer (BaseNormalizer): The fitted normalizer for decision data.
            - metrics (Dict): A dictionary of calculated validation metrics.
        """
        # Split data into train and validation sets
        objectives_train, objectives_val, decisions_train, decisions_val = (
            train_test_split(
                objectives, decisions, test_size=test_size, random_state=random_state
            )
        )

        # Normalize training and validation data
        objectives_train = self._objectives_normalizer.fit_transform(objectives_train)
        objectives_val = self._objectives_normalizer.transform(objectives_val)
        decisions_train = self._decisions_normalizer.fit_transform(decisions_train)

        # Fit the interpolator instance on normalized data
        inverse_decision_mapper.fit(
            objectives=objectives_train,  # Objectives are the input to the inverse mapper
            decisions=decisions_train,  # Decisions are the output of the inverse mapper
        )

        # Predict decision values on the validation set
        decisions_pred_val = inverse_decision_mapper.predict(objectives_val)

        # Inverse-transform predictions to original scale
        decisions_pred_val_2_original = self._decisions_normalizer.inverse_transform(
            decisions_pred_val
        )

        # Calculate validation metrics using the injected metric
        metrics = {
            self._validation_metric.name: self._validation_metric.calculate(
                y_true=decisions_val, y_pred=decisions_pred_val_2_original
            )
        }

        print(f"Validation Metrics: {metrics}")

        # Return fitted instance, fitted normalizers, and metrics
        return inverse_decision_mapper, metrics

    def predict(
        self,
        fitted_inverse_decision_mapper: BaseInverseDecisionMapper,
        target_objectives_norm: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Generates predictions using a fitted interpolator and inverse-transforms them.

        Args:
            fitted_inverse_decision_mapper: A pre-fitted instance of BaseInverseDecisionMapper.
            target_objectives_norm: Normalized objective values for which to make predictions.

        Returns:
            predicted_decisions (NDArray): Predicted decision values in their original scale.
        """
        # Predict normalized values (these are normalized decision values)
        predicted_decisions_norm = fitted_inverse_decision_mapper.predict(
            target_objectives_norm
        )

        # Inverse-transform predictions to original scale
        predicted_decisions = self._decisions_normalizer.inverse_transform(
            predicted_decisions_norm
        )

        return predicted_decisions
