from ...application.datasets.process_dataset.process_dataset_command import (
    ProcessDatasetCommand,
)
from ...application.datasets.process_dataset.process_dataset_handler import (
    ProcessDatasetCommandHandler,
)
from ...application.dtos import NormalizerConfig
from ...application.factories.normalizer import NormalizerFactory
from ...infrastructure.datasets.repositories.generated_dataset_repo import (
    FileSystemGeneratedDatasetRepository,
)
from ...infrastructure.datasets.repositories.processed_dataset_repo import (
    FileSystemProcessedDatasetRepository,
)
from ...infrastructure.loggers.cmd_logger import CMDLogger

handler = ProcessDatasetCommandHandler(
    generated_dataset_repo=FileSystemGeneratedDatasetRepository(),
    processed_dataset_repo=FileSystemProcessedDatasetRepository(),
    normalizer_factory=NormalizerFactory(),
    logger=CMDLogger(),
)

cmd = ProcessDatasetCommand(
    source_filename="dataset",
    dest_filename="processed_v1",
    normalizer_config=NormalizerConfig(type="hypercube", params={}),
    test_size=0.2,
    random_state=42,
    include_original=True,
    overwrite=True,
    dataset_source="historical",
)

# Execute
handler.execute(cmd)
