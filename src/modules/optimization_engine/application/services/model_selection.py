import math
import warnings
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import KFold

from ...domain.model_evaluation.interfaces.base_validation_metric import (
    BaseValidationMetric,
)
from ...domain.model_management.interfaces.base_estimator import (
    BaseEstimator,
    DeterministicEstimator,
    ProbabilisticEstimator,
)


class EpochsCurveService:
    """Collects per-epoch train/val losses from a probabilistic estimator."""

    def run(
        self,
        estimator: ProbabilisticEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_val: np.typing.NDArray,
        y_val: np.typing.NDArray,
        epochs: int = 100,
        batch_size: int = 32,
        plot: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(estimator, ProbabilisticEstimator):
            raise TypeError("EpochsCurveService is for ProbabilisticEstimator only")

        # Fit with validation data to get a per-epoch validation history
        try:
            estimator.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
            )
        except TypeError:
            warnings.warn(
                "ProbabilisticEstimator does not support 'validation_data' argument. Fitting without."
            )
            estimator.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        history = None
        try:
            history = estimator.get_loss_history()
        except Exception:
            history = None

        if history is None:
            return {
                "bin_type": "epoch",
                "bins": list(range(epochs)),
                "train_loss": [None] * epochs,
                "val_loss": [None] * epochs,
                "test_loss": [None] * epochs,
            }

        epochs_bins = list(
            history.get("epochs", list(range(len(history.get("train_loss", [])))))
        )
        train_loss = [
            float(x) if x is not None else None for x in history.get("train_loss", [])
        ]
        val_loss = [
            float(x) if x is not None else None for x in history.get("val_loss", [])
        ]

        result = {
            "bin_type": "epoch",
            "bins": epochs_bins,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": [None] * len(train_loss),
        }

        if plot:
            self._show_plot(
                bins=epochs_bins,
                train_loss=train_loss,
                val_loss=val_loss,
                xlabel="Epoch",
                ylabel="Loss",
                title="Epochs Curve",
            )

        return result

    @staticmethod
    def _show_plot(
        bins: List[Any],
        train_loss: List[float],
        val_loss: List[float],
        xlabel: str,
        ylabel: str,
        title: str,
    ) -> None:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=bins, y=train_loss, mode="lines+markers", name="Train Loss")
        )
        if any(v is not None for v in val_loss):
            fig.add_trace(
                go.Scatter(
                    x=bins, y=val_loss, mode="lines+markers", name="Validation Loss"
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            hovermode="x unified",
            template="plotly_white",
        )
        fig.show()


class LearningCurveService:
    """Builds learning-curve style training history by subsampling training set and cloning the estimator."""

    def run(
        self,
        estimator: DeterministicEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_test: np.typing.NDArray,
        y_test: np.typing.NDArray,
        metrics: Dict[str, BaseValidationMetric],
        learning_curve_steps: int = 50,
        random_state: int = 0,
        plot: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(estimator, DeterministicEstimator):
            raise TypeError("LearningCurveService is for DeterministicEstimator only")

        if metrics is None or len(metrics) == 0:
            raise ValueError("metrics dict must be provided and non-empty")

        metric_names = list(metrics.keys())
        loss_metric_name = "MSE" if "MSE" in metric_names else metric_names[0]

        n_total = len(X_train)
        fractions = list(np.linspace(0.1, 1.0, learning_curve_steps))
        rng = np.random.RandomState(random_state)

        bins: List[float] = []
        n_train_list: List[int] = []
        train_loss_list: List[float] = []
        val_loss_list: List[float] = []
        test_loss_list: List[float] = []
        print(f"Building learning curve with {learning_curve_steps} steps...")
        for frac in fractions:
            n = max(1, int(math.floor(frac * n_total)))
            bins.append(float(frac))
            n_train_list.append(int(n))

            idx = rng.choice(np.arange(n_total), size=n, replace=False)
            X_sub = X_train[idx]
            y_sub = y_train[idx]

            cloned = estimator.clone()
            cloned.fit(X_sub, y_sub)

            t_loss = self._evaluate_single_metric(
                cloned, X_sub, y_sub, metrics, loss_metric_name
            )
            train_loss_list.append(t_loss)

            ts_loss = self._evaluate_single_metric(
                cloned, X_test, y_test, metrics, loss_metric_name
            )
            test_loss_list.append(ts_loss)

        result = {
            "bin_type": "train_fraction",
            "bins": bins,
            "n_train": n_train_list,
            "train_loss": train_loss_list,
            "val_loss": [],
            "test_loss": test_loss_list,
        }

        if plot:
            self._show_plot(
                n_train_list,
                train_loss_list,
                val_loss_list,
                test_loss_list,
                loss_metric_name,
                "Learning Curve",
            )
        return result

    @staticmethod
    def _evaluate_point_metrics(
        estimator: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        metrics: Dict[str, BaseValidationMetric],
    ) -> Dict[str, float]:
        if X is None or len(X) == 0:
            return {name: float("nan") for name in metrics}

        if isinstance(estimator, ProbabilisticEstimator):
            y_pred = estimator.predict(X, mode="mean")
        else:
            y_pred = estimator.predict(X)

        results: Dict[str, float] = {}
        for name, metric in metrics.items():
            try:
                results[name] = float(metric.calculate(y_true=y, y_pred=y_pred))
            except Exception:
                results[name] = float("nan")
        return results

    def _evaluate_single_metric(
        self,
        estimator: BaseEstimator,
        X: np.typing.NDArray,
        y: np.typing.NDArray,
        metrics: Dict[str, BaseValidationMetric],
        metric_name: str,
    ) -> float:
        if X is None or len(X) == 0:
            return None
        try:
            scores = self._evaluate_point_metrics(estimator, X, y, metrics)
            return float(scores.get(metric_name, np.nan))
        except Exception:
            return None

    @staticmethod
    def _show_plot(
        n_train: List[int],
        train_loss: List[float],
        val_loss: List[float],
        test_loss: List[float],
        ylabel: str,
        title: str,
    ) -> None:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=n_train, y=train_loss, mode="lines+markers", name="Train")
        )
        if any(v is not None for v in val_loss):
            fig.add_trace(
                go.Scatter(
                    x=n_train, y=val_loss, mode="lines+markers", name="Validation"
                )
            )
        if any(v is not None for v in test_loss):
            fig.add_trace(
                go.Scatter(x=n_train, y=test_loss, mode="lines+markers", name="Test")
            )

        fig.update_layout(
            title=title,
            xaxis_title="Number of Training Samples",
            yaxis_title=ylabel,
            hovermode="x unified",
            template="plotly_white",
        )
        fig.show()


class ValidationCurveService:
    """Runs a validation curve: sweep a parameter across a range and compute train/validation scores."""

    def _set_param(self, estimator: BaseEstimator, param_name: str, value: Any) -> None:
        if hasattr(estimator, "set_params"):
            try:
                estimator.set_params(**{param_name: value})
                return
            except Exception:
                pass
        if hasattr(estimator, "params") and isinstance(
            getattr(estimator, "params"), dict
        ):
            estimator.params[param_name] = value
            return
        attr_name = param_name
        if hasattr(estimator, attr_name):
            try:
                setattr(estimator, attr_name, value)
                return
            except Exception:
                pass
        try:
            estimator.__dict__[param_name] = value
            return
        except Exception:
            return

    def run(
        self,
        estimator: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        param_name: str,
        param_range: Iterable[Any],
        *,
        cv: int = 5,
        metrics: Dict[str, BaseValidationMetric],
        random_state: int = 44,
        plot: bool = False,
    ) -> dict[str, Any]:
        param_vals = list(param_range)
        train_scores_all: Dict[str, list] = {name: [] for name in metrics.keys()}
        valid_scores_all: Dict[str, list] = {name: [] for name in metrics.keys()}

        kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)

        for val in param_vals:
            fold_train_scores: Dict[str, list] = {name: [] for name in metrics.keys()}
            fold_valid_scores: Dict[str, list] = {name: [] for name in metrics.keys()}

            for train_idx, val_idx in kf.split(X):
                X_tr, X_va = X[train_idx], X[val_idx]
                y_tr, y_va = y[train_idx], y[val_idx]

                cloned = estimator.clone()
                self._set_param(cloned, param_name, val)

                if isinstance(cloned, ProbabilisticEstimator):
                    cloned.fit(X_tr, y_tr)
                else:
                    cloned.fit(X_tr, y_tr)

                y_tr_pred = (
                    cloned.predict(X_tr, mode="mean")
                    if isinstance(cloned, ProbabilisticEstimator)
                    else cloned.predict(X_tr)
                )
                y_va_pred = (
                    cloned.predict(X_va, mode="mean")
                    if isinstance(cloned, ProbabilisticEstimator)
                    else cloned.predict(X_va)
                )

                for mname, metric in metrics.items():
                    try:
                        fold_train_scores[mname].append(
                            float(metric.calculate(y_true=y_tr, y_pred=y_tr_pred))
                        )
                    except Exception:
                        fold_train_scores[mname].append(float("nan"))
                    try:
                        fold_valid_scores[mname].append(
                            float(metric.calculate(y_true=y_va, y_pred=y_va_pred))
                        )
                    except Exception:
                        fold_valid_scores[mname].append(float("nan"))

            for mname in metrics.keys():
                train_scores_all[mname].append(
                    float(np.nanmean(fold_train_scores[mname]))
                )
                valid_scores_all[mname].append(
                    float(np.nanmean(fold_valid_scores[mname]))
                )

        result = {
            "param_name": param_name,
            "param_range": param_vals,
            "train_scores": train_scores_all,
            "valid_scores": valid_scores_all,
        }

        if plot:
            self._show_plot(
                param_vals, param_name, train_scores_all, valid_scores_all, metrics
            )

        return result

    @staticmethod
    def _show_plot(
        param_vals: list,
        param_name: str,
        train_scores_all: Dict[str, list],
        valid_scores_all: Dict[str, list],
        metrics: Dict[str, Any],
    ) -> None:
        fig = make_subplots(
            rows=1, cols=1, subplot_titles=[f"Validation Curve for {param_name}"]
        )

        for mname in metrics.keys():
            fig.add_trace(
                go.Scatter(
                    x=param_vals,
                    y=train_scores_all[mname],
                    mode="lines+markers",
                    name=f"Train ({mname})",
                    legendgroup="train",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=param_vals,
                    y=valid_scores_all[mname],
                    mode="lines+markers",
                    name=f"Validation ({mname})",
                    legendgroup="valid",
                ),
                row=1,
                col=1,
            )

        fig.update_xaxes(title_text=param_name, row=1, col=1)
        fig.update_yaxes(title_text="Score", row=1, col=1)
        if all(isinstance(x, (int, float)) and x > 0 for x in param_vals):
            fig.update_xaxes(type="log", row=1, col=1)

        fig.update_layout(
            title="Validation Curve",
            hovermode="x unified",
            template="plotly_white",
        )
        fig.show()
