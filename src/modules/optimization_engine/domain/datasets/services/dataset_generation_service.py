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
        *,
        dataset_name: str,
        optimizer: BaseOptimizer,
        X_normalizer: BaseNormalizer,
        y_normalizer: BaseNormalizer,
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
            X=opt_data.historical_solutions,
            y=opt_data.historical_objectives,
            pareto=pareto,
        )

        processed_dataset = self._build_processed_dataset(
            raw_dataset=generated_dataset,
            X_normalizer=X_normalizer,
            y_normalizer=y_normalizer,
            test_size=test_size,
            random_state=random_state,
            metadata=dict(metadata or {}),
        )

        return generated_dataset, processed_dataset

    def _build_processed_dataset(
        self,
        raw_dataset: GeneratedDataset,
        X_normalizer: BaseNormalizer,
        y_normalizer: BaseNormalizer,
        test_size: float,
        random_state: int,
        metadata: dict[str, Any],
    ) -> ProcessedDataset:
        """Split, normalize, and package processed data artifacts."""

        X_raw = raw_dataset.pareto.set
        y_raw = raw_dataset.pareto.front

        X_train, X_test, y_train, y_test = train_test_split(
            X_raw,
            y_raw,
            test_size=test_size,
            random_state=random_state,
        )

        X_train_norm = X_normalizer.fit_transform(X_train)
        X_test_norm = X_normalizer.transform(X_test)
        y_train_norm = y_normalizer.fit_transform(y_train)
        y_test_norm = y_normalizer.transform(y_test)

        normalized_pareto = None
        if raw_dataset.pareto is not None:
            normalized_pareto = Pareto.create(
                set=X_normalizer.transform(raw_dataset.pareto.set),
                front=y_normalizer.transform(raw_dataset.pareto.front),
            )

        metadata.update(
            {
                "test_size": test_size,
                "random_state": random_state,
            }
        )

        return ProcessedDataset.create(
            name=raw_dataset.name,
            X_train=X_train_norm,
            y_train=y_train_norm,
            X_test=X_test_norm,
            y_test=y_test_norm,
            X_normalizer=X_normalizer,
            y_normalizer=y_normalizer,
            pareto=normalized_pareto,
            metadata=metadata,
        )
