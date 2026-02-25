import numpy as np
from sklearn.neighbors import NearestNeighbors

from ....dataset.domain.interfaces.base_repository import BaseDatasetRepository
from ....modeling.domain.interfaces.base_repository import BaseTrainedPipelineRepository
from ....shared.domain.interfaces.base_logger import BaseLogger
from ...domain.entities.coherence_context import CoherenceContext
from ...domain.interfaces.base_context_repository import BaseContextRepository


class PrepareContextService:
    """
    Orchestrates the offline context preparation for generation:
    1. Loads dataset
    2. Identifies correct surrogate (TrainedPipeline)
    3. Transforms data into normalized space
    4. Computes coherence threshold (tau)
    5. Persists the CoherenceContext
    """

    def __init__(
        self,
        dataset_repository: BaseDatasetRepository,
        model_repository: BaseTrainedPipelineRepository,
        context_repository: BaseContextRepository,
        logger: BaseLogger,
    ):
        self._dataset_repository = dataset_repository
        self._model_repository = model_repository
        self._context_repository = context_repository
        self._logger = logger

    def execute(
        self, dataset_name: str, surrogate_type: str, k_neighbors: int
    ) -> CoherenceContext:
        self._logger.log_info(
            f"Preparing coherence context for '{dataset_name}' with surrogate '{surrogate_type}'"
        )

        # 1. Load dataset (now we just need raw data)
        dataset = self._dataset_repository.load(dataset_name)

        # 2. Get surrogate pipeline
        pipeline = self._model_repository.get_latest_version(
            estimator_type=surrogate_type,
            mapping_direction="forward",
            dataset_name=dataset_name,
        )

        # 3. Get raw data
        objectives = dataset.objectives
        decisions = dataset.decisions

        # 4. Normalize the decisions using pipeline transforms to get anchors_norm
        anchors_norm = decisions.copy()
        for t in pipeline.get_decisions_transforms():
            anchors_norm = t.transform(anchors_norm)

        # We also need to normalize objectives for the context?
        # Wait, the CoherenceContext is built on the *normalized decisions* (anchors_norm)
        # But `target_objective` during generation might need normalization.
        # The context stores `objectives`. We can store raw or normalized.
        # Old code: `objectives = np.vstack([dataset.processed.y_train, ...])`.
        # In old code, `processed.y_train` for FORWARD model (where X=decisions, y=objectives)
        # But wait, `dataset.processed.y_train` for inverse model is decisions... wait no.
        # In prepare_context.py, old code had: `objectives = dataset.processed.y_train` (wait, is that objectives or decisions?)
        # Let's think: `objectives` variable in Context is used to find nearest anchor in Objective space.
        # So it should be `objectives`. We will store raw objectives and normalize at generation time.

        # 5. Compute tau
        nn = NearestNeighbors(n_neighbors=k_neighbors + 1)  # +1 to include self
        nn.fit(anchors_norm)
        distances, _ = nn.kneighbors(anchors_norm)

        # distances[:, 1:] ignores the zero-distance to self
        tau = float(np.percentile(distances[:, 1:], 95))
        self._logger.log_info(
            f"Computed tau: {tau:.4f} using k={k_neighbors} (95th percentile)"
        )

        # 6. Create and persist Context
        context = CoherenceContext(
            dataset_name=dataset_name,
            objectives=objectives,
            anchors_norm=anchors_norm,
            tau=tau,
            k_neighbors=k_neighbors,
            surrogate_type=surrogate_type,
            surrogate_version=pipeline.version,
        )

        self._context_repository.save(context)
        self._logger.log_info("CoherenceContext saved successfully.")

        return context
