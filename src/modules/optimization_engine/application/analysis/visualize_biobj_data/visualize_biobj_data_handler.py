import numpy as np

from ....domain.analysis.interfaces.base_visualizer import BaseDataVisualizer
from ....domain.generation.interfaces.base_repository import BaseParetoDataRepository
from ...factories.normalizer import NormalizerFactory
from .visualize_biobj_data_command import VisualizeBiobjDataCommand


class VisulizeBiobjDataCommandHandler:
    """
    Loads a bi-objective dataset, builds normalizers via an injected factory,
    normalizes decisions (X) and objectives (F), and passes everything to the
    visualizer as a compact dict.

    """

    def __init__(
        self,
        data_repo: BaseParetoDataRepository,
        visualizer: BaseDataVisualizer,
        normalizer_factory: NormalizerFactory,
    ):
        self._data_repo = data_repo
        self._visualizer = visualizer
        self._normalizer_factory = normalizer_factory

    def execute(self, command: VisualizeBiobjDataCommand) -> None:
        # 1) Load raw data from repository
        raw = self._data_repo.load(filename=command.data_file_name)

        pareto_set = np.asarray(getattr(raw, "pareto_set", None))  # (n, 2)
        pareto_front = np.asarray(getattr(raw, "pareto_front", None))  # (n, 2)
        hist_solutions = getattr(raw, "historical_solutions", None)  # optional (m, 2)
        hist_objectives = getattr(raw, "historical_objectives", None)  # optional (k, 2)

        # 3) Build distinct normalizers for each space via the injected factory
        paret_set_normalizer = self._normalizer_factory.create(
            command.normalizer_config.model_dump()
        )
        pareto_front_normalizer = self._normalizer_factory.create(
            command.normalizer_config.model_dump()
        )

        # 4) Fit + transform (one per space). Keep them 2D as (n, 2).
        pareto_set_norm = paret_set_normalizer.fit_transform(pareto_set)  # (n, 2)
        pareto_front_norm = pareto_front_normalizer.fit_transform(
            pareto_front
        )  # (n, 2)

        historical_solutions_norm = paret_set_normalizer.transform(hist_solutions)

        historical_objectives_norm = pareto_front_normalizer.transform(hist_objectives)

        # 6) Package payload (arrays only)
        payload = {
            "pareto_set": pareto_set,
            "pareto_front": pareto_front,
            "pareto_set_norm": pareto_set_norm,
            "pareto_front_norm": pareto_front_norm,
            "historical_solutions": hist_solutions,
            "historical_objectives": hist_objectives,
            "historical_solutions_norm": historical_solutions_norm,
            "historical_objectives_norm": historical_objectives_norm,
        }

        # 7) Plot
        self._visualizer.plot(data=payload)
