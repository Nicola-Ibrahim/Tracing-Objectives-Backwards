from sklearn.model_selection import train_test_split

from ....domain.common.interfaces.base_logger import BaseLogger
from ....domain.datasets.entities.processed_dataset import ProcessedDataset
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ....domain.modeling.interfaces.base_normalizer import BaseNormalizer
from ...factories.normalizer import NormalizerFactory
from .process_dataset_command import ProcessDatasetCommand


class ProcessDatasetCommandHandler:
    """
    Reads raw Pareto data, performs a train/test split, fits normalizers on train,
    transforms train/test, and saves a processed dataset bundle.
    """

    def __init__(
        self,
        generated_dataset_repo: BaseDatasetRepository,
        processed_dataset_repo: BaseDatasetRepository,
        normalizer_factory: NormalizerFactory,
        logger: BaseLogger,
    ) -> None:
        self._generated_dataset_repo = generated_dataset_repo
        self._processed_dataset_repo = processed_dataset_repo
        self._normalizer_factory = normalizer_factory
        self._logger = logger

    def execute(self, command: ProcessDatasetCommand) -> None:
        # 1) Load raw bundle
        generated_dataset = self._generated_dataset_repo.load(
            filename=command.source_filename
        )
        self._logger.log_info(
            f"[postprocess] data: X{generated_dataset.X.shape}, y{generated_dataset.y.shape}"
        )

        # 2) Split FIRST (best practice to avoid leakage)
        #    Note: dataset uses (objectives=y, decisions=X).
        X_raw = generated_dataset.y  # objectives
        y_raw = generated_dataset.X  # solutions/decisions

        X_train, X_test, y_train, y_test = train_test_split(
            X_raw, y_raw, test_size=command.test_size, random_state=command.random_state
        )

        # 3) Build normalizers (one per space)
        X_normalizer: BaseNormalizer = self._normalizer_factory.create(
            command.normalizer_config.model_dump()
        )
        y_normalizer: BaseNormalizer = self._normalizer_factory.create(
            command.normalizer_config.model_dump()
        )

        # 4) Fit on TRAIN only; transform TRAIN and TEST
        X_train = X_normalizer.fit_transform(X_train)
        X_test = X_normalizer.transform(X_test)
        y_train = y_normalizer.fit_transform(y_train)
        y_test = y_normalizer.transform(y_test)

        # 5) Bundle + save
        processed_dataset = ProcessedDataset.create(
            name="processed_dataset",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            X_normalizer=X_normalizer,  # fitted instances
            y_normalizer=y_normalizer,  # fitted instances
            pareto=generated_dataset.pareto,
            metadata={
                "source": command.source_filename,
                "test_size": command.test_size,
                "random_state": command.random_state,
                "normalizer": command.normalizer_config.model_dump(),
            },
        )
        self._processed_dataset_repo.save(processed_dataset)
        self._logger.log_info(
            f"[postprocess] saved processed dataset to '{command.dest_filename}.pkl'"
        )
