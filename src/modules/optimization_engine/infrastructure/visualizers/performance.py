from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

from ...domain.modeling.interfaces.base_estimator import (
    BaseEstimator,
    DeterministicEstimator,
    ProbabilisticEstimator,
)
from ...domain.visualization.interfaces.base_visualizer import BaseVisualizer

try:
    from umap import UMAP

    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False


class ModelPerformanceVisualizer(BaseVisualizer):
    """
    Visualize model fits on *normalized* data + residual diagnostics.

    Expected `data` keys (all ALREADY NORMALIZED):
      - estimator
      - X_train, y_train
      - X_test (optional), y_test (optional)
      - n_samples (int, default 200)
      - non_linear (bool, default False)  # for dimensionality reduction of X to 1D
      - title (str, optional)
      - loss_history (dict)     # e.g. outcome.loss_history.model_dump()
    """

    # ------------------------------- public ------------------------------- #

    @dataclass
    class _Payload:
        estimator: Any
        X_train: np.ndarray
        y_train: np.ndarray
        X_test: np.typing.NDArray | None
        y_test: np.typing.NDArray | None
        n_samples: int
        non_linear: bool
        title: str
        loss_history: Any

    @dataclass
    class _Predictions:
        train_pred: np.ndarray
        train_resid: np.ndarray
        test_pred: np.typing.NDArray | None
        test_resid: np.typing.NDArray | None

    @dataclass
    class _ReducedInputs:
        reducer: Any
        train: np.typing.NDArray | None
        test: np.typing.NDArray | None

    def plot(self, data: Any) -> None:
        payload, dims = self._prepare_payload(data)
        predictions = self._compute_predictions(payload)

        x_dim, y_dim = dims
        if x_dim == 2 and y_dim == 2:
            self._plot_2d_case(payload, predictions)
            return

        else:
            reduced = self._reduce_inputs(payload, x_dim, y_dim)
            for out_idx in range(y_dim):
                fig = self._build_general_figure(payload.title, out_idx)
                y_train = payload.y_train[:, out_idx : out_idx + 1]
                y_test = (
                    payload.y_test[:, out_idx : out_idx + 1]
                    if payload.y_test is not None
                    else None
                )

                self._add_fit_row_general(
                    fig=fig,
                    estimator=payload.estimator,
                    X=payload.X_train,
                    X_reduced=reduced.train,
                    y_train=y_train,
                    n_samples=payload.n_samples,
                    out_idx=out_idx,
                )
                self._overlay_training_testing_points(
                    fig=fig,
                    X_train_reduced=reduced.train,
                    X_test_reduced=reduced.test,
                    y_train=y_train,
                    y_test=y_test,
                )

                self._add_loss_curves_row(fig, row=2, loss_history=payload.loss_history)

                self._add_residual_panels(
                    fig=fig,
                    predictions=predictions,
                    out_idx=out_idx,
                )

                fig.update_xaxes(
                    title_text="Objective (reduced, normalized)", row=1, col=1
                )
                fig.update_yaxes(title_text="Decision (normalized)", row=1, col=1)
                self._add_estimator_summary(fig, payload.estimator)
                fig.update_layout(
                    template="plotly_white",
                    height=1100,
                    margin=dict(l=60, r=240, t=80, b=60),
                )
                fig.show()

    # ----------------------------- preparation ----------------------------- #

    def _prepare_payload(
        self, data: Any
    ) -> tuple["ModelPerformanceVisualizer._Payload", tuple[int, int]]:
        if not isinstance(data, dict):
            raise TypeError("ModelPerformanceVisualizer expects `data` to be a dict.")

        estimator = data["estimator"]
        X_train = np.asarray(data["X_train"])
        y_train = np.asarray(data["y_train"])
        X_test = np.asarray(data["X_test"]) if data.get("X_test") is not None else None
        y_test = np.asarray(data["y_test"]) if data.get("y_test") is not None else None

        if data.get("loss_history") is None:
            raise ValueError(
                "ModelPerformanceVisualizer requires `loss_history` in data."
            )

        payload = self._Payload(
            estimator=estimator,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_samples=int(data.get("n_samples", 200)),
            non_linear=bool(data.get("non_linear", False)),
            title=data.get("title", f"Model fit ({type(estimator).__name__})"),
            loss_history=data["loss_history"],
        )

        x_dim = X_train.shape[1]
        y_dim = y_train.shape[1]
        return payload, (x_dim, y_dim)

    def _compute_predictions(
        self, payload: "ModelPerformanceVisualizer._Payload"
    ) -> "ModelPerformanceVisualizer._Predictions":
        train_pred = self._predict_pointwise(
            payload.estimator, payload.X_train, payload.n_samples
        )
        train_resid = payload.y_train - train_pred

        if payload.X_test is not None and payload.y_test is not None:
            test_pred = self._predict_pointwise(
                payload.estimator, payload.X_test, payload.n_samples
            )
            test_resid = payload.y_test - test_pred
        else:
            test_pred = None
            test_resid = None

        return self._Predictions(
            train_pred=train_pred,
            train_resid=train_resid,
            test_pred=test_pred,
            test_resid=test_resid,
        )

    def _reduce_inputs(
        self,
        payload: "ModelPerformanceVisualizer._Payload",
        x_dim: int,
        y_dim: int,
    ) -> "ModelPerformanceVisualizer._ReducedInputs":
        if x_dim == 2 and y_dim == 2:
            return self._ReducedInputs(reducer=None, train=None, test=None)

        reducer, X_train_reduced = self._fit_reducer_to_1d(
            payload.X_train, payload.non_linear
        )
        if payload.X_test is None:
            X_test_reduced = None
        elif reducer is None:
            X_test_reduced = (
                payload.X_test[:, 0] if payload.X_test.shape[1] == 1 else None
            )
        else:
            X_test_reduced = self._transform_reducer(reducer, payload.X_test)

        return self._ReducedInputs(
            reducer=reducer,
            train=X_train_reduced,
            test=X_test_reduced,
        )

    # ------------------------------- general ------------------------------- #

    def _build_general_figure(self, title: str, out_idx: int) -> go.Figure:
        return make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.12,
            subplot_titles=(
                f"{title} — y{out_idx} (normalized)",
                "Training / Validation / Test",
                "Residuals vs Fitted (train + test)",
                "Residual distribution (train + test)",
            ),
        )

    def _add_fit_row_general(
        self,
        *,
        fig: go.Figure,
        estimator: Any,
        X: np.ndarray,
        X_reduced: np.typing.NDArray | None,
        y_train: np.ndarray,
        n_samples: int,
        out_idx: int,
    ) -> None:
        if isinstance(estimator, ProbabilisticEstimator):
            self._add_prob_fit_row(
                fig,
                row=1,
                estimator=estimator,
                X_red=X_reduced,
                X=X,
                y_raw_1d=y_train,
                n_samples=n_samples,
            )
        else:
            self._add_det_fit_row(
                fig,
                row=1,
                estimator=estimator,
                X_red=X_reduced,
                X=X,
                y_raw_1d=y_train,
            )

    def _overlay_training_testing_points(
        self,
        *,
        fig: go.Figure,
        X_train_reduced: np.typing.NDArray | None,
        X_test_reduced: np.typing.NDArray | None,
        y_train: np.ndarray,
        y_test: np.typing.NDArray | None,
    ) -> None:
        if X_train_reduced is None:
            return
        fig.add_trace(
            go.Scatter(
                x=X_train_reduced,
                y=y_train[:, 0],
                mode="markers",
                name="Train",
                marker=dict(opacity=0.35),
            ),
            row=1,
            col=1,
        )
        if X_test_reduced is not None and y_test is not None:
            fig.add_trace(
                go.Scatter(
                    x=X_test_reduced,
                    y=y_test[:, 0],
                    mode="markers",
                    name="Test",
                    marker=dict(opacity=0.35),
                ),
                row=1,
                col=1,
            )

    def _add_residual_panels(
        self,
        *,
        fig: go.Figure,
        predictions: "ModelPerformanceVisualizer._Predictions",
        out_idx: int,
    ) -> None:
        y_pred_train = predictions.train_pred[:, out_idx]
        resid_train = predictions.train_resid[:, out_idx]

        if predictions.test_pred is not None and predictions.test_resid is not None:
            y_pred_test = predictions.test_pred[:, out_idx]
            resid_test = predictions.test_resid[:, out_idx]
            self._add_residuals_vs_fitted(
                fig,
                row=3,
                fitted=y_pred_test,
                resid=resid_test,
                output_name=f"y{out_idx} (test)",
            )
        self._add_residuals_vs_fitted(
            fig,
            row=3,
            fitted=y_pred_train,
            resid=resid_train,
            output_name=f"y{out_idx} (train)",
        )

        if predictions.test_resid is not None:
            self._add_residual_hist(
                fig,
                row=4,
                resid=predictions.test_resid[:, out_idx],
                output_name=f"y{out_idx} (test)",
            )
        self._add_residual_hist(
            fig,
            row=4,
            resid=resid_train,
            output_name=f"y{out_idx} (train)",
        )

    # ---------------------- 2D→2D (surfaces + panels) --------------------- #

    def _plot_2d_case(
        self,
        payload: "ModelPerformanceVisualizer._Payload",
        predictions: "ModelPerformanceVisualizer._Predictions",
        grid_res: int = 50,
    ) -> None:
        """
        2D→2D case: render two 3D surfaces y1(x1,x2) and y2(x1,x2) plus
        training/validation curves and residual diagnostics. Axes across both
        3D scenes are forced to share the same x/y/z ranges and aspect.
        """

        estimator = payload.estimator
        Xtr = payload.X_train
        Ytr = payload.y_train
        Xte = payload.X_test
        Yte = payload.y_test
        title = payload.title
        loss_history = payload.loss_history

        y_pred_test = predictions.test_pred
        resid_test = predictions.test_resid
        y_pred_train = predictions.train_pred
        resid_train = predictions.train_resid

        # --- build grid over TRAIN objective domain (x1,x2) ---
        x1_min, x1_max = float(np.min(Xtr[:, 0])), float(np.max(Xtr[:, 0]))
        x2_min, x2_max = float(np.min(Xtr[:, 1])), float(np.max(Xtr[:, 1]))
        gx1 = np.linspace(x1_min, x1_max, grid_res)
        gx2 = np.linspace(x2_min, x2_max, grid_res)
        GX1, GX2 = np.meshgrid(gx1, gx2, indexing="xy")
        X_grid = np.stack([GX1.ravel(), GX2.ravel()], axis=1)

        # predict Y on the X-grid (model maps x -> y). Returns (n, 2) for y1,y2
        Yg = self._predict_pointwise(estimator, X_grid, payload.n_samples)
        z_y1 = Yg[:, 0].reshape(grid_res, grid_res)
        z_y2 = Yg[:, 1].reshape(grid_res, grid_res)

        fig = make_subplots(
            rows=5,
            cols=2,
            specs=[
                [{"type": "surface"}, {"type": "surface"}],  # row1: surfaces
                [{"type": "xy", "colspan": 2}, None],  # row2: curves
                [{"type": "xy"}, {"type": "xy"}],  # row3: residuals vs fitted
                [{"type": "xy"}, {"type": "xy"}],  # row4: residual hists
                [{"type": "xy", "colspan": 2}, None],  # row5: joint resid dist
            ],
            subplot_titles=[
                "Objectives (x1,x2) → Decision y1 (normalized)",
                "Objectives (x1,x2) → Decision y2 (normalized)",
                "Training / Validation / Test",
                "Residuals vs Fitted (decision y1)",
                "Residuals vs Fitted (decision y2)",
                "Residual distribution (decision y1)",
                "Residual distribution (decision y2)",
                "Residual joint distribution (y1 vs y2)",
            ],
            vertical_spacing=0.06,
            horizontal_spacing=0.07,
            row_heights=[0.42, 0.14, 0.18, 0.18, 0.08],
        )

        # -------- row1: surfaces + TRAIN/TEST point clouds (x on axes; y as height) -----
        fig.add_trace(
            go.Surface(
                x=GX1,
                y=GX2,
                z=z_y1,
                opacity=0.45,
                showscale=False,
                name="Decision y1(x)",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Surface(
                x=GX1,
                y=GX2,
                z=z_y2,
                opacity=0.45,
                showscale=False,
                name="Decision y2(x)",
            ),
            row=1,
            col=2,
        )

        # TRAIN points: (x1,x2) on plane; z = y1 / y2
        fig.add_trace(
            go.Scatter3d(
                x=Xtr[:, 0],
                y=Xtr[:, 1],
                z=Ytr[:, 0],
                mode="markers",
                name="Train decisions (y1)",
                marker=dict(size=3, opacity=0.5),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=Xtr[:, 0],
                y=Xtr[:, 1],
                z=Ytr[:, 1],
                mode="markers",
                name="Train decisions (y2)",
                marker=dict(size=3, opacity=0.5),
            ),
            row=1,
            col=2,
        )

        # TEST points (if provided)
        if Xte is not None and Yte is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=Xte[:, 0],
                    y=Xte[:, 1],
                    z=Yte[:, 0],
                    mode="markers",
                    name="Test decisions (y1)",
                    marker=dict(size=3, opacity=0.5),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter3d(
                    x=Xte[:, 0],
                    y=Xte[:, 1],
                    z=Yte[:, 1],
                    mode="markers",
                    name="Test decisions (y2)",
                    marker=dict(size=3, opacity=0.5),
                ),
                row=1,
                col=2,
            )

        # ------------------ row2..row5: diagnostics -------------------
        self._add_loss_curves_row(fig, row=2, loss_history=loss_history, col=1)

        if y_pred_test is not None and resid_test is not None:
            self._add_residuals_vs_fitted(
                fig,
                row=3,
                col=1,
                fitted=y_pred_test[:, 0],
                resid=resid_test[:, 0],
                output_name="decision y1 (test)",
            )
            self._add_residuals_vs_fitted(
                fig,
                row=3,
                col=2,
                fitted=y_pred_test[:, 1],
                resid=resid_test[:, 1],
                output_name="decision y2 (test)",
            )
        self._add_residuals_vs_fitted(
            fig,
            row=3,
            col=1,
            fitted=y_pred_train[:, 0],
            resid=resid_train[:, 0],
            output_name="decision y1 (train)",
        )
        self._add_residuals_vs_fitted(
            fig,
            row=3,
            col=2,
            fitted=y_pred_train[:, 1],
            resid=resid_train[:, 1],
            output_name="decision y2 (train)",
        )

        if resid_test is not None:
            self._add_residual_hist(
                fig,
                row=4,
                col=1,
                resid=resid_test[:, 0],
                output_name="decision y1 (test)",
            )
            self._add_residual_hist(
                fig,
                row=4,
                col=2,
                resid=resid_test[:, 1],
                output_name="decision y2 (test)",
            )
        self._add_residual_hist(
            fig,
            row=4,
            col=1,
            resid=resid_train[:, 0],
            output_name="decision y1 (train)",
        )
        self._add_residual_hist(
            fig,
            row=4,
            col=2,
            resid=resid_train[:, 1],
            output_name="decision y2 (train)",
        )

        if resid_test is not None:
            self._add_joint_residual_distribution(
                fig, row=5, col=1, resid_y1=resid_test[:, 0], resid_y2=resid_test[:, 1]
            )

        # --- unified axis ranges & consistent aspect/ticks for both 3D scenes ---
        def _minmax(*arrs):
            vals = [np.asarray(a).ravel() for a in arrs if a is not None]
            if not vals:
                return (0.0, 1.0)
            v = np.concatenate(vals)
            v = v[np.isfinite(v)]
            return (float(v.min()), float(v.max())) if v.size else (0.0, 1.0)

        # common XY ranges taken from X (domain); include the grid and any test Xs
        x_rng = _minmax(Xtr[:, 0], (Xte[:, 0] if Xte is not None else None), GX1)
        y_rng = _minmax(Xtr[:, 1], (Xte[:, 1] if Xte is not None else None), GX2)

        # Z ranges from Y (targets) + predicted surfaces
        z1_min, z1_max = _minmax(
            Ytr[:, 0], (Yte[:, 0] if Yte is not None else None), z_y1
        )
        z2_min, z2_max = _minmax(
            Ytr[:, 1], (Yte[:, 1] if Yte is not None else None), z_y2
        )
        z_rng = (min(z1_min, z2_min), max(z1_max, z2_max))

        # small padding so points aren’t glued to box
        def _pad(a, b, frac=0.02):
            d = (b - a) if b > a else 1.0
            return (a - frac * d, b + frac * d)

        x_rng = _pad(*x_rng)
        y_rng = _pad(*y_rng)
        z_rng = _pad(*z_rng)

        # If your data are strictly normalized and you want hard clamp to [0,1], use:
        # x_rng = (0.0, 1.0); y_rng = (0.0, 1.0); z_rng = (0.0, 1.0)

        # build identical tick locations so both scenes show the same ticks
        def _ticks(rng, n=5):
            return list(np.round(np.linspace(rng[0], rng[1], n), 5))

        scene1_axes = dict(
            xaxis=dict(
                title="Objective x1 (norm)",
                range=list(x_rng),
                tickmode="array",
                tickvals=_ticks(x_rng),
            ),
            yaxis=dict(
                title="Objective x2 (norm)",
                range=list(y_rng),
                tickmode="array",
                tickvals=_ticks(y_rng),
            ),
            zaxis=dict(
                title="Decision y1 (norm)",
                range=list(z_rng),
                tickmode="array",
                tickvals=_ticks(z_rng),
            ),
            aspectmode="cube",  # equal scale, no distortion
        )
        scene2_axes = dict(
            xaxis=dict(
                title="Objective x1 (norm)",
                range=list(x_rng),
                tickmode="array",
                tickvals=_ticks(x_rng),
            ),
            yaxis=dict(
                title="Objective x2 (norm)",
                range=list(y_rng),
                tickmode="array",
                tickvals=_ticks(y_rng),
            ),
            zaxis=dict(
                title="Decision y2 (norm)",
                range=list(z_rng),
                tickmode="array",
                tickvals=_ticks(z_rng),
            ),
            aspectmode="cube",
        )

        # final layout + apply scene settings
        fig.update_layout(
            title=title + " — fit, curves & residuals (normalized)",
            template="plotly_white",
            height=1600,
            width=1600,
            margin=dict(l=60, r=280, t=80, b=60),
        )
        self._add_estimator_summary(fig, estimator)
        fig.update_scenes(row=1, col=1, **scene1_axes)
        fig.update_scenes(row=1, col=2, **scene2_axes)

        fig.show()

    # -------------------------- residual panels --------------------------- #

    def _add_residuals_vs_fitted(
        self,
        fig: go.Figure,
        *,
        row: int,
        fitted: np.ndarray,
        resid: np.ndarray,
        output_name: str,
        col: int = 1,
    ) -> None:
        fig.add_trace(
            go.Scatter(
                x=fitted,
                y=resid,
                mode="markers",
                name=f"Residuals ({output_name})",
                marker=dict(opacity=0.35, size=6),
            ),
            row=row,
            col=col,
        )
        # zero line as a 2D trace (avoid add_hline/shapes)
        if np.size(fitted) > 0:
            x_min = float(np.nanmin(fitted))
            x_max = float(np.nanmax(fitted))
            if np.isfinite(x_min) and np.isfinite(x_max) and x_min != x_max:
                fig.add_trace(
                    go.Scatter(
                        x=[x_min, x_max],
                        y=[0.0, 0.0],
                        mode="lines",
                        name="Zero residual",
                        line=dict(color="black", width=1),
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )
        fig.update_xaxes(title_text="Fitted (norm)", row=row, col=col)
        fig.update_yaxes(title_text="Residual (norm)", row=row, col=col)

    def _add_residual_hist(
        self,
        fig: go.Figure,
        *,
        row: int,
        resid: np.ndarray,
        output_name: str,
        col: int = 1,
    ) -> None:
        fig.add_trace(
            go.Histogram(
                x=resid,
                nbinsx=40,
                histnorm="probability density",
                name=f"Resid ({output_name})",
                opacity=0.7,
            ),
            row=row,
            col=col,
        )
        vals = resid[np.isfinite(resid)]
        if vals.size:
            counts, _ = np.histogram(vals, bins=40, density=True)
            ymax = float(np.nanmax(counts)) if counts.size else 1.0
            fig.add_trace(
                go.Scatter(
                    x=[0.0, 0.0],
                    y=[0.0, ymax * 1.05],
                    mode="lines",
                    name="Zero",
                    line=dict(color="black", width=1),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(title_text="Residual (norm)", row=row, col=col)
        fig.update_yaxes(title_text="Density", row=row, col=col)

    def _add_joint_residual_distribution(
        self,
        fig: go.Figure,
        *,
        row: int,
        col: int,
        resid_y1: np.ndarray,
        resid_y2: np.ndarray,
    ) -> None:
        vals_x = resid_y1[np.isfinite(resid_y1)]
        vals_y = resid_y2[np.isfinite(resid_y2)]
        fig.add_trace(
            go.Histogram2dContour(
                x=vals_x,
                y=vals_y,
                ncontours=20,
                contours_coloring="heatmap",
                showscale=False,
                name="Residual density",
                opacity=0.85,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=vals_x,
                y=vals_y,
                mode="markers",
                name="Residuals",
                marker=dict(size=4, opacity=0.25),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        if vals_x.size and vals_y.size:
            x_min, x_max = float(np.nanmin(vals_x)), float(np.nanmax(vals_x))
            y_min, y_max = float(np.nanmin(vals_y)), float(np.nanmax(vals_y))
            if np.isfinite(x_min) and np.isfinite(x_max) and x_min != x_max:
                fig.add_trace(
                    go.Scatter(
                        x=[x_min, x_max],
                        y=[0.0, 0.0],
                        mode="lines",
                        line=dict(color="black", width=1),
                        name="y2=0",
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )
            if np.isfinite(y_min) and np.isfinite(y_max) and y_min != y_max:
                fig.add_trace(
                    go.Scatter(
                        x=[0.0, 0.0],
                        y=[y_min, y_max],
                        mode="lines",
                        line=dict(color="black", width=1),
                        name="y1=0",
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )
        fig.update_xaxes(title_text="Residual y1 (norm)", row=row, col=col)
        fig.update_yaxes(title_text="Residual y2 (norm)", row=row, col=col)

    # ----------------------- loss curves helper row ----------------------- #

    def _add_loss_curves_row(
        self,
        fig: go.Figure,
        row: int,
        loss_history: dict[str, Any] | None,
        col: int = 1,
    ) -> None:
        if loss_history is None:
            return
        # Allow pydantic value-objects (e.g. LossHistory) as well as raw dicts
        if hasattr(loss_history, "model_dump"):
            loss_history = loss_history.model_dump()
        elif hasattr(loss_history, "dict"):
            loss_history = loss_history.dict()
        if not isinstance(loss_history, dict):
            return
        bins: list[float] = list(loss_history.get("bins", []))
        train_loss: list[float] = list(loss_history.get("train_loss", []))
        val_loss: list[float] = list(loss_history.get("val_loss", []))
        test_loss: list[float] = list(loss_history.get("test_loss", []))
        n_train: list[int] = list(loss_history.get("n_train", []))
        bin_type: str = str(loss_history.get("bin_type", "bin"))
        # Choose x-axis and labels
        if bin_type == "train_fraction" and n_train:
            x_vals = n_train
            x_label = "Number of Training Samples"
        elif bin_type == "epoch":
            x_vals = bins if bins else list(range(1, len(train_loss) + 1))
            x_label = "Epoch"
        else:
            x_vals = bins if bins else list(range(len(train_loss)))
            x_label = {"param": "Parameter", "bin": "Bin"}.get(
                bin_type, bin_type.title()
            )
        if train_loss:
            fig.add_trace(
                go.Scatter(x=x_vals, y=train_loss, mode="lines+markers", name="Train"),
                row=row,
                col=col,
            )
        if any(v is not None for v in val_loss):
            fig.add_trace(
                go.Scatter(
                    x=x_vals, y=val_loss, mode="lines+markers", name="Validation"
                ),
                row=row,
                col=col,
            )
        if any(v is not None for v in test_loss):
            fig.add_trace(
                go.Scatter(x=x_vals, y=test_loss, mode="lines+markers", name="Test"),
                row=row,
                col=col,
            )
        fig.update_xaxes(title_text=x_label, row=row, col=col)
        fig.update_yaxes(title_text="Loss / Score", row=row, col=col)

    def _add_estimator_summary(self, fig: go.Figure, estimator: BaseEstimator) -> None:
        """Add estimator hyperparameters as text annotation on the figure."""
        if not hasattr(estimator, "to_dict"):
            return
        params = estimator.to_dict()
        if not isinstance(params, dict) or not params:
            return

        lines = ["Estimator parameters:"]
        for key, value in params.items():
            lines.append(f"{key}: {value}")
        summary = "<br>".join(lines)

        fig.update_layout(
            legend=dict(
                x=1.02,
                y=1.0,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
                traceorder="normal",
            )
        )

        line_count = max(len(lines), 1)
        # Position the summary below the legend regardless of its size.
        summary_top = max(0.0, 1.0 - 0.1 - 0.04 * (line_count - 1))

        fig.add_annotation(
            text=summary,
            align="left",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=1.02,
            y=summary_top,
            xanchor="left",
            yanchor="top",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            borderpad=6,
            bgcolor="rgba(255,255,255,0.9)",
            font=dict(size=11),
        )

    # ------------------------- 1D fit-row helpers ------------------------ #

    def _add_det_fit_row(
        self,
        fig: go.Figure,
        row: int,
        *,
        estimator,
        X_red,
        X,
        y_raw_1d,
    ):
        y_pred = np.atleast_2d(estimator.predict(X))  # normalized
        order = np.argsort(X_red)
        fig.add_trace(
            go.Scatter(
                x=X_red[order], y=y_pred[order, 0], mode="lines", name="Prediction"
            ),
            row=row,
            col=1,
        )

    def _add_prob_fit_row(
        self,
        fig: go.Figure,
        row: int,
        *,
        estimator,
        X_red,
        X,
        y_raw_1d,
        n_samples: int,
    ):
        samples = estimator.sample(X, n_samples=max(2, n_samples))
        arr = np.asarray(samples)
        if arr.ndim == 2:
            arr = arr[:, None, :]
        elif arr.ndim == 3 and arr.shape[1] != max(2, n_samples):
            arr = np.transpose(arr, (1, 0, 2))

        try:
            map_pred = estimator.predict_map(X)
            if map_pred.ndim == 1:
                map_pred = map_pred[:, None]
            center_line = map_pred[:, 0]
        except Exception:
            center_line = arr.mean(axis=1)[:, 0]

        p05 = np.percentile(arr, 5, axis=1)[:, 0]
        p95 = np.percentile(arr, 95, axis=1)[:, 0]
        order = np.argsort(X_red)
        fig.add_trace(
            go.Scatter(x=X_red[order], y=center_line[order], mode="lines", name="MAP"),
            row=row,
            col=1,
        )
        ribbon_x = np.concatenate([X_red[order], X_red[order][::-1]])
        ribbon_y = np.concatenate([p95[order], p05[order][::-1]])
        fig.add_trace(
            go.Scatter(
                x=ribbon_x,
                y=ribbon_y,
                fill="toself",
                fillcolor="rgba(0,100,200,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="5–95%",
            ),
            row=row,
            col=1,
        )

    def _add_mdn_fit_row(
        self,
        fig: go.Figure,
        row: int,
        *,
        estimator,
        X,
        X_red,
        y_raw_1d,
        out_idx,
    ):
        mp = self._get_mdn_params(estimator, X)
        pi, mu, sigma = mp["pi"], mp["mu"], mp["sigma"]
        if mu.ndim == 2:
            mu = mu[..., None]
        n_pts, K, _ = mu.shape
        mu_k = mu[:, :, out_idx]  # normalized
        order = np.argsort(X_red)
        base_colors = [
            "rgba(31,119,180,1.0)",
            "rgba(214,39,40,1.0)",
            "rgba(44,160,44,1.0)",
            "rgba(148,103,189,1.0)",
            "rgba(255,127,14,1.0)",
            "rgba(23,190,207,1.0)",
        ]
        z = 1.96
        for k in range(K):
            comp_mu = mu_k[:, k][order]
            fig.add_trace(
                go.Scatter(
                    x=X_red[order],
                    y=comp_mu,
                    mode="lines",
                    name=f"Component {k+1}",
                    line=dict(color=base_colors[k % len(base_colors)], width=3),
                ),
                row=row,
                col=1,
            )
            if sigma is not None:
                if sigma.ndim == 2:
                    sigma = sigma[..., None]
                sig_k = sigma[:, :, out_idx]
                comp_sig = np.clip(sig_k[:, k], 1e-12, None)[order]
                upper = comp_mu + z * comp_sig
                lower = comp_mu - z * comp_sig
                ribbon_x = np.concatenate([X_red[order], X_red[order][::-1]])
                ribbon_y = np.concatenate([upper, lower[::-1]])
                alpha = float(np.clip(np.mean(pi[:, k]), 0.15, 0.6))
                rgba = base_colors[k % len(base_colors)].replace("1.0", f"{alpha:.3f}")
                fig.add_trace(
                    go.Scatter(
                        x=ribbon_x,
                        y=ribbon_y,
                        fill="toself",
                        fillcolor=rgba,
                        line=dict(color="rgba(255,255,255,0)"),
                        name=f"Comp {k+1} band",
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=1,
                )

    # ----------------------------- utilities ----------------------------- #

    def _fit_reducer_to_1d(self, X: np.ndarray, non_linear: bool):
        """Fit a 1D reducer on TRAIN and return (reducer, X_train_reduced)."""
        if X.shape[1] == 1:
            return None, X[:, 0]
        if non_linear and _HAS_UMAP:
            reducer = UMAP(n_components=1)
            Xr = reducer.fit_transform(X).squeeze()
            return reducer, Xr
        reducer = PCA(n_components=1)
        Xr = reducer.fit_transform(X).squeeze()
        return reducer, Xr

    def _transform_reducer(
        self, reducer, X: np.typing.NDArray | None
    ) -> np.typing.NDArray | None:
        if X is None:
            return None
        if reducer is None:
            # 1D case handled by caller
            return X[:, 0] if X.shape[1] == 1 else None
        xr = reducer.transform(X).squeeze()
        return xr

    def _predict_pointwise(
        self,
        estimator: Any,
        X: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """Return mean prediction in normalized space, shape (n, y_dim)."""
        if isinstance(estimator, ProbabilisticEstimator):
            return estimator.predict_mean(X, n_samples=max(2, n_samples))

        if isinstance(estimator, DeterministicEstimator):
            return estimator.predict(X)

        raise TypeError("Unsupported estimator type for prediction.")
