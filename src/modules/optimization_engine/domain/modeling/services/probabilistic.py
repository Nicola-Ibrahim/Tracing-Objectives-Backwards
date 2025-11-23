import numpy as np

from ..interfaces.base_estimator import (
    BaseEstimator,
    ProbabilisticEstimator,
)
from ..value_objects.loss_history import LossHistory
from ..value_objects.metrics import Metrics


class ProbabilisticModelTrainer:
    """
    Train probabilistic estimators and return TrainingOutcome.
    """

    def compute_epoch_history(
        self,
        estimator: ProbabilisticEstimator,
        X_train: np.typing.NDArray,
        y_train: np.typing.NDArray,
        tandem: tuple[BaseEstimator, float] | None = None,
    ) -> tuple[BaseEstimator, LossHistory]:
        """
        (helper) Fit estimator and extract per-epoch loss history if provided by estimator.
        """

        if not isinstance(estimator, ProbabilisticEstimator):
            raise TypeError("EpochsCurve is for ProbabilisticEstimator only")

        # 1) Fit estimator for given number of epochs (estimator should internally record history)

        estimator.fit(X_train, y_train, tandem=tandem)

        # 2) Try to read history from common accessor names
        history = estimator.get_loss_history()

        # 3) Build LossHistory from recorded history
        bins = history.get("epochs", [])
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])

        loss_history = LossHistory(
            bin_type="epoch",
            bins=bins,
            n_train=[len(X_train)] * len(bins),
            train_loss=train_loss,
            val_loss=val_loss,
            test_loss=[],
        )

        return estimator, loss_history

    def train(
        self,
        estimator: BaseEstimator,
        X_train: np.typing.NDArray,
        y_train: np.typing.NDArray,
        epochs: int = 50,
        batch_size: int = 64,
        tandem: tuple[BaseEstimator, float] | None = None,
    ) -> tuple[BaseEstimator, LossHistory, Metrics]:
        """
        Public entry to train a probabilistic estimator on raw (X, y).
        Step-by-step:
          1) Split raw data into train/test
          2) Normalize train and test using provided normalizers
          3) Fit the estimator for a number of epochs and extract epoch history
          4) Return tuple with fitted estimator, loss_history and normalized arrays
        """

        # 1) fit and collect epoch history
        if not isinstance(estimator, ProbabilisticEstimator):
            raise TypeError("ProbabilisticModelTrainer requires ProbabilisticEstimator")

        estimator, loss_history = self.compute_epoch_history(
            estimator=estimator,
            X_train=X_train,
            y_train=y_train,
            tandem=tandem,
        )

        # 2) produce TrainingOutcome; leave train/test point-scores None (caller can compute if desired)
        metrics = Metrics(train=[], test=[], cv=[])

        return estimator, loss_history, metrics
