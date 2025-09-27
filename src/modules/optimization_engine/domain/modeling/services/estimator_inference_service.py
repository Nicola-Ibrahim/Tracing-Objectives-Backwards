import numpy as np

from ..interfaces.base_estimator import BaseEstimator, ProbabilisticEstimator


class EstimatorInferenceService:
    """Generate raw and normalized outputs from an estimator given normalized inputs."""

    def infer(
        self,
        *,
        estimator: BaseEstimator,
        X_norm: np.ndarray,
        normalizer,
        n_samples: int | None = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return denormalized and normalized outputs for the provided inputs."""

        X_input = np.atleast_2d(np.asarray(X_norm, dtype=float))

        if isinstance(estimator, ProbabilisticEstimator):
            sample_count = int(n_samples) if n_samples is not None else 256
            y_norm = estimator.predict_mean(X_input, n_samples=sample_count)
        else:
            y_norm = estimator.predict(X_input)

        y_norm = np.atleast_2d(np.asarray(y_norm, dtype=float))
        if y_norm.shape[0] != X_input.shape[0]:
            raise ValueError(
                "Estimator returned mismatched batch size during inference."
            )

        y_raw = normalizer.inverse_transform(y_norm)[0]
        return y_raw, y_norm[0]
