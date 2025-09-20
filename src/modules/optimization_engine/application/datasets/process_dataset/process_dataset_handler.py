import numpy as np

from ....domain.common.interfaces.base_logger import BaseLogger
from ....domain.datasets.entities.processed_dataset import ProcessedDataset
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ...factories.normalizer import NormalizerFactory
from ...services.utils import split_and_normalize
from .process_dataset_command import ProcessDatasetCommand


class ProcessDatasetCommandHandler:
    """
    Reads raw Pareto data, performs a train/test split with normalization,
    and saves the processed dataset into the processed repository as a pickle.
    """

    def __init__(
        self,
        raw_repo: BaseDatasetRepository,
        processed_repo: BaseDatasetRepository,
        normalizer_factory: NormalizerFactory,
        logger: BaseLogger,
    ) -> None:
        self._raw_repo = raw_repo
        self._processed_repo = processed_repo
        self._normalizer_factory = normalizer_factory
        self._logger = logger

    def execute(self, command: ProcessDatasetCommand) -> None:
        # 1) Load raw bundle
        raw = self._raw_repo.load(filename=command.source_filename)

        self._logger.log_info(
            f"[postprocess] data: X{raw.historical_objectives.shape}, y{raw.historical_solutions.shape}"
        )

        # 2) Build normalizers (one per space)
        X_normalizer = self._normalizer_factory.create(
            command.normalizer_config.model_dump()
        )
        y_normalizer = self._normalizer_factory.create(
            command.normalizer_config.model_dump()
        )

        # 3) Split + normalize (returns normalized arrays + the fitted normalizers)
        X_train, X_test, y_train, y_test, X_normalizer_fitted, y_normalizer_fitted = (
            split_and_normalize(
                X=raw.historical_objectives,
                y=raw.historical_solutions,
                X_normalizer=X_normalizer,
                y_normalizer=y_normalizer,
                test_size=command.test_size,
                random_state=command.random_state,
            )
        )

        # 4) Bundle output
        prcessed_dataset = ProcessedDataset.create(
            name="processed_dataset",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            X_normalizer=X_normalizer_fitted,
            y_normalizer=y_normalizer_fitted,
            pareto_set=raw.pareto_set,
            pareto_front=raw.pareto_front,
            metadata={
                "source": command.source_filename,
                "test_size": command.test_size,
                "random_state": command.random_state,
                "normalizer": command.normalizer_config.model_dump(),
            },
        )

        # 5) Save to processed repository
        self._processed_repo.save(prcessed_dataset)

        self._logger.log_info(
            f"[postprocess] saved processed dataset to '{command.dest_filename}.pkl'"
        )
