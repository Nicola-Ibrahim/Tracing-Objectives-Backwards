import numpy as np

from ....domain.interpolation.interfaces.base_repository import (
    BaseInterpolationModelRepository,
)
from .free_mode_generate_decision_command import FreeModeGenerateDecisionCommand


class FreeModeGenerateDecisionCommandHandler:
    """
    Handles the command to generate a decision (input parameters)
    for a given target objective in free mode.
    """

    def __init__(
        self,
        interpolation_model_repo: BaseInterpolationModelRepository,
    ):
        """
        Initializes the handler with the necessary repository.
        Normalizers are now loaded directly from the model artifacts.

        Args:
            interpolation_model_repo: Repository for accessing interpolation models.
        """
        self._interpolation_model_repo = interpolation_model_repo

    def handle(self, command: FreeModeGenerateDecisionCommand) -> np.ndarray:
        """
        Fetches the latest trained model of a specific type, uses the stored
        fitted normalizers to preprocess the target objective, generates the
        predicted decision, and then denormalizes it.

        Args:
            command: The command containing the interpolator type and target objective.

        Returns:
            An np.ndarray representing the predicted decision in the original scale.

        Raises:
            FileNotFoundError: If no model of the specified type is found.
            Exception: For other errors during model loading or prediction.
        """
        # Fetch the latest trained model based on the interpolator_type
        # This now loads the fitted model AND its associated fitted normalizers
        model = self._interpolation_model_repo.get_latest_version(
            interpolator_type=command.interpolator_type
        )
        # Use the interpolator and normalizers loaded with the model
        inverse_decision_mapper = model.inverse_decision_mapper
        decisions_normalizer = model.decisions_normalizer

        # Normalize the target objective using the loaded, fitted normalizer
        y_norm = decisions_normalizer.transform(np.array(command.target_objective))

        # Generate the predicted decision in the normalized space
        x_pred_norm = inverse_decision_mapper.predict(y_norm)[0]

        # Denormalize the predicted decision back to its original scale
        x_pred = decisions_normalizer.inverse_transform(np.array([x_pred_norm]))[0]

        return x_pred
