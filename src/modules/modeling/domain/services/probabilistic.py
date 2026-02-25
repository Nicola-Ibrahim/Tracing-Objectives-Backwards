import numpy as np

from ..interfaces.base_estimator import (
    BaseEstimator,
    ProbabilisticEstimator,
)
from ..value_objects.estimator_step import TrainingLog
from ..value_objects.evaluation_result import EvaluationResult


class ProbabilisticModelTrainer:
    """
    Train probabilistic estimators and return TrainingOutcome.
    """

    def compute_epoch_history(
        self,
        estimator: ProbabilisticEstimator,
        X_train: np.typing.NDArray,
        y_train: np.typing.NDArray,
    ) -> tuple[BaseEstimator, dict[str, list[float]]]:
        """
        (helper) Fit estimator and extract per-epoch loss history from estimator.
        """

        if not isinstance(estimator, ProbabilisticEstimator):
            raise TypeError("EpochsCurve is for ProbabilisticEstimator only")

        # 1) Fit estimator for given number of epochs (estimator internally records history)
        estimator.fit(X_train, y_train)

        # 2) Read history from estimator
        training_history = estimator.get_loss_history()

        return estimator, training_history

    def train(
        self,
        estimator: BaseEstimator,
        X_train: np.typing.NDArray,
        y_train: np.typing.NDArray,
        epochs: int = 50,
        batch_size: int = 64,
    ) -> tuple[BaseEstimator, TrainingLog, EvaluationResult]:
        """
        Public entry to train a probabilistic estimator on raw (X, y).
        """

        # 1) fit and collect epoch history
        if not isinstance(estimator, ProbabilisticEstimator):
            raise TypeError("ProbabilisticModelTrainer requires ProbabilisticEstimator")

        estimator, training_history = self.compute_epoch_history(
            estimator=estimator,
            X_train=X_train,
            y_train=y_train,
        )

        # 2) produce TrainingOutcome; leave train/test point-scores None (caller can compute if desired)
        metrics = EvaluationResult(train=[], test=[], cv=[])

        return estimator, TrainingLog(**training_history), metrics
