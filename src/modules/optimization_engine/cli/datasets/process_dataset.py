from ...application.datasets.process_dataset.process_dataset_command import (
    ProcessDatasetCommand,
)
from ...application.datasets.process_dataset.process_dataset_handler import (
    ProcessDatasetCommandHandler,
)
from ...application.dtos import NormalizerConfig
from ...application.factories.normalizer import NormalizerFactory
from ...infrastructure.loggers.cmd_logger import CMDLogger
from ...infrastructure.repositories.datasets.generated_dataset_repo import (
    FileSystemGeneratedDatasetRepository,
)
from ...infrastructure.repositories.datasets.processed_dataset_repo import (
    FileSystemProcessedDatasetRepository,
)

# Repositories
raw_repo = FileSystemGeneratedDatasetRepository()
processed_repo = FileSystemProcessedDatasetRepository()

# Handler (DI)
handler = ProcessDatasetCommandHandler(
    raw_repo=raw_repo,
    processed_repo=processed_repo,
    normalizer_factory=NormalizerFactory(),
    logger=CMDLogger(),
)

# Command
cmd = ProcessDatasetCommand(
    source_filename="dataset",
    dest_filename="processed_v1",
    normalizer_config=NormalizerConfig(
        type="MinMaxScaler", params={"feature_range": (0, 1)}
    ),
    test_size=0.2,
    random_state=42,
    include_original=True,
    overwrite=True,
    dataset_source="historical",
)

# Execute
handler.execute(cmd)
