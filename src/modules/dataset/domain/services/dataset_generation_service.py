from typing import Any

from ..entities.dataset import Dataset
from ..interfaces.base_data_source import BaseDataSource
from ..value_objects.pareto import Pareto


class DatasetGenerationService:
    """Generate and preprocess datasets from any configured data source."""

    def generate(
        self,
        dataset_name: str,
        data_source: BaseDataSource,
        metadata: dict[str, Any] | None = None,
    ) -> Dataset:
        """Fetch raw data from source and build the Dataset aggregate."""

        raw = data_source.generate()

        # Pareto is optional — only created when the data source provides it
        pareto = None
        if raw.pareto_set is not None and raw.pareto_front is not None:
            pareto = Pareto.create(
                set=raw.pareto_set,
                front=raw.pareto_front,
            )

        dataset = Dataset.create(
            name=dataset_name,
            decisions=raw.decisions,
            objectives=raw.objectives,
            pareto=pareto,
        )

        return dataset
