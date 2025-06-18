import numpy as np
from sklearn.neighbors import NearestNeighbors

from ....domain.models.interfaces.base_normalizer import BaseNormalizer
from ....domain.models.interfaces.base_repository import (
    BaseTrainedModelRepository,
)
from ....domain.paretos.interfaces.base_archiver import BaseParetoArchiver
from .guided_mode_generate_decision_command import GuidedModeGenerateDecisionCommand


class GuidedModeGenerateDecisionHandler:
    def __init__(
        self,
        model_repository: BaseTrainedModelRepository,
        pareto_archiver: BaseParetoArchiver,
        y_normalizer: BaseNormalizer,
        x_normalizer: BaseNormalizer,
    ):
        self._model_repository = model_repository
        self._pareto_archiver = pareto_archiver
        self._y_normalizer = y_normalizer
        self._x_normalizer = x_normalizer

    def handle(self, command: GuidedModeGenerateDecisionCommand) -> np.ndarray:
        # Load model
        model = self._model_repository.load(command.interpolator_name)
        interpolator = model.fitted_interpolator

        # Load data
        data = self._pareto_archiver.load(model.source_path)
        X, Y = data.pareto_set, data.pareto_front

        # Normalize
        self._y_normalizer.fit(Y)
        self._x_normalizer.fit(X)

        y_norm_all = self._y_normalizer.transform(Y)
        x_norm_all = self._x_normalizer.transform(X)
        y_target_norm = self._y_normalizer.transform([command.target_objective])

        # Find neighbors
        nn = NearestNeighbors(n_neighbors=command.neighbor_count)
        nn.fit(y_norm_all)
        _, indices = nn.kneighbors(y_target_norm)

        # Fit on neighbors
        X_neighbors = x_norm_all[indices[0]]
        Y_neighbors = y_norm_all[indices[0]]
        interpolator.fit(X_neighbors, Y_neighbors)

        # Generate and denormalize
        x_pred_norm = interpolator.generate(y_target_norm)[0]
        x_pred = self._x_normalizer.inverse_transform([x_pred_norm])[0]
        return x_pred
