import numpy as np

from ....domain.interpolation.interfaces.base_normalizer import BaseNormalizer
from ....domain.interpolation.interfaces.base_repository import (
    BaseInterpolationModelRepository,
)
from .free_mode_generate_decision_command import FreeModeGenerateDecisionCommand


class FreeModeGenerateDecisionHandler:
    def __init__(
        self,
        model_repository: BaseInterpolationModelRepository,
        y_normalizer: BaseNormalizer,
        x_normalizer: BaseNormalizer,
    ):
        self._model_repository = model_repository
        self._y_normalizer = y_normalizer
        self._x_normalizer = x_normalizer

    def handle(self, command: FreeModeGenerateDecisionCommand) -> np.ndarray:
        model = self._model_repository.load(command.interpolator_name)
        interpolator = model.fitted_interpolator

        y_train = model.y_train
        self._y_normalizer.fit(y_train)
        self._x_normalizer.fit(model.x_train)

        y_norm = self._y_normalizer.transform([command.target_objective])
        x_pred_norm = interpolator.generate(y_norm)[0]
        x_pred = self._x_normalizer.inverse_transform([x_pred_norm])[0]

        return x_pred
