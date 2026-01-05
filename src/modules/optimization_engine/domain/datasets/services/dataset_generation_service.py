from typing import Any

from sklearn.model_selection import train_test_split

from ...datasets.entities.dataset import Dataset
from ...datasets.entities.processed_data import ProcessedData
from ...datasets.interfaces.base_optimizer import BaseOptimizer
from ...datasets.value_objects.pareto import Pareto
from ...modeling.interfaces.base_normalizer import BaseNormalizer


class DatasetGenerationService:
    """Generate and preprocess datasets from configured optimization components."""

    def generate(
        self,
        dataset_name: str,
        optimizer: BaseOptimizer,
        decisions_normalizer: BaseNormalizer,
        objectives_normalizer: BaseNormalizer,
        test_size: float,
        random_state: int,
        metadata: dict[str, Any] | None = None,
    ) -> Dataset:
        """Run optimization and build dataset artifacts."""

        opt_data = optimizer.run()

        pareto = Pareto.create(
            set=opt_data.pareto_set,
            front=opt_data.pareto_front,
        )

        dataset = Dataset.create(
            name=dataset_name,
            decisions=opt_data.historical_solutions,
            objectives=opt_data.historical_objectives,
            pareto=pareto,
        )

        processed_data = self._build_processed_data(
            dataset=dataset,
            decisions_normalizer=decisions_normalizer,
            objectives_normalizer=objectives_normalizer,
            test_size=test_size,
            random_state=random_state,
            metadata=dict(metadata or {}),
        )

        dataset.add_processed_visuals(processed_data)

        return dataset

    def _build_processed_data(
        self,
        dataset: Dataset,
        decisions_normalizer: BaseNormalizer,
        objectives_normalizer: BaseNormalizer,
        test_size: float,
        random_state: int,
        metadata: dict[str, Any],
    ) -> ProcessedData:
        """Split, normalize, and package processed data artifacts."""

        decisions_raw = dataset.decisions
        objectives_raw = dataset.objectives

        decisions_train, decisions_test, objectives_train, objectives_test = (
            train_test_split(
                decisions_raw,
                objectives_raw,
                test_size=test_size,
                random_state=random_state,
            )
        )

        decisions_train_norm = decisions_normalizer.fit_transform(decisions_train)
        decisions_test_norm = decisions_normalizer.transform(decisions_test)
        objectives_train_norm = objectives_normalizer.fit_transform(objectives_train)
        objectives_test_norm = objectives_normalizer.transform(objectives_test)

        # Note: Normalized pareto is not stored in ProcessedData in this design,
        # but can be derived. If it was needed separately, we could add it back.
        # Original ProcessedDataset had a pareto field. Let's check where it's used.
        # But ProcessedData entity didn't include pareto field in my design.
        # I should double check if I missed it in ProcessedData.

        metadata.update(
            {
                "test_size": test_size,
                "random_state": random_state,
            }
        )

        return ProcessedData.create(
            decisions_train=decisions_train_norm,
            objectives_train=objectives_train_norm,
            decisions_test=decisions_test_norm,
            objectives_test=objectives_test_norm,
            decisions_normalizer=decisions_normalizer,
            objectives_normalizer=objectives_normalizer,
            metadata=metadata,
        )
