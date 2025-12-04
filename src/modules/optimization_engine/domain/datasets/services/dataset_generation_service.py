from typing import Any

from sklearn.model_selection import train_test_split

from ...datasets.entities.generated_dataset import GeneratedDataset
from ...datasets.entities.processed_dataset import ProcessedDataset
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
    ) -> tuple[GeneratedDataset, ProcessedDataset]:
        """Run optimization and build both raw and processed dataset artifacts."""

        opt_data = optimizer.run()

        pareto = Pareto.create(
            set=opt_data.pareto_set,
            front=opt_data.pareto_front,
        )

        generated_dataset = GeneratedDataset(
            name=dataset_name,
            decisions=opt_data.historical_solutions,
            objectives=opt_data.historical_objectives,
            pareto=pareto,
        )
        print(opt_data.historical_solutions.shape)
        print(opt_data.historical_objectives.shape)

        processed_dataset = self._build_processed_dataset(
            raw_dataset=generated_dataset,
            decisions_normalizer=decisions_normalizer,
            objectives_normalizer=objectives_normalizer,
            test_size=test_size,
            random_state=random_state,
            metadata=dict(metadata or {}),
        )

        return generated_dataset, processed_dataset

    def _build_processed_dataset(
        self,
        raw_dataset: GeneratedDataset,
        decisions_normalizer: BaseNormalizer,
        objectives_normalizer: BaseNormalizer,
        test_size: float,
        random_state: int,
        metadata: dict[str, Any],
    ) -> ProcessedDataset:
        """Split, normalize, and package processed data artifacts."""

        decisions_raw = raw_dataset.decisions
        objectives_raw = raw_dataset.objectives

        decisions_train, decisions_test, objectives_train, objectives_test = train_test_split(
            decisions_raw,
            objectives_raw,
            test_size=test_size,
            random_state=random_state,
        )

        decisions_train_norm = decisions_normalizer.fit_transform(decisions_train)
        decisions_test_norm = decisions_normalizer.transform(decisions_test)
        objectives_train_norm = objectives_normalizer.fit_transform(objectives_train)
        objectives_test_norm = objectives_normalizer.transform(objectives_test)

        normalized_pareto = None
        if raw_dataset.pareto is not None:
            normalized_pareto = Pareto.create(
                set=decisions_normalizer.transform(raw_dataset.pareto.set),
                front=objectives_normalizer.transform(raw_dataset.pareto.front),
            )

        metadata.update(
            {
                "test_size": test_size,
                "random_state": random_state,
            }
        )

        return ProcessedDataset.create(
            name=raw_dataset.name,
            decisions_train=decisions_train_norm,
            objectives_train=objectives_train_norm,
            decisions_test=decisions_test_norm,
            objectives_test=objectives_test_norm,
            decisions_normalizer=decisions_normalizer,
            objectives_normalizer=objectives_normalizer,
            pareto=normalized_pareto,
            metadata=metadata,
        )
