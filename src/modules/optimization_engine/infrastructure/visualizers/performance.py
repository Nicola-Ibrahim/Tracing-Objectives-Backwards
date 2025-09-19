from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

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
      - loss_history (dict, optional)     # e.g. outcome.loss_history.model_dump()
      - show_train_residuals (bool, default False)
    """

    # ------------------------------- public ------------------------------- #

    def plot(self, data: Any) -> None:
        if not isinstance(data, dict):
            raise TypeError("ModelPerformanceVisualizer expects `data` to be a dict.")

        est = data["estimator"]
        Xtr = np.asarray(data["X_train"])
        ytr = np.asarray(data["y_train"])
        Xte = np.asarray(data["X_test"]) if data.get("X_test") is not None else None
        yte = np.asarray(data["y_test"]) if data.get("y_test") is not None else None

        n_samples = int(data.get("n_samples", 200))
        non_linear = bool(data.get("non_linear", False))
        title = data.get("title", f"Model fit ({type(est).__name__})")
        loss_history: Optional[Dict[str, Any]] = data.get("loss_history")
        show_train_residuals: bool = bool(data.get("show_train_residuals", False))

        ntr, x_dim = Xtr.shape
        _, y_dim = ytr.shape

        # Fit reducer on TRAIN ONLY; transform TRAIN (and TEST if provided)
        reducer, Xtr_red = (
            self._fit_reducer_to_1d(Xtr, non_linear)
            if not (x_dim == 2 and y_dim == 2)
            else (None, None)
        )
        Xte_red = (
            self._transform_reducer(reducer, Xte)
            if (reducer is not None and Xte is not None)
            else (
                Xte[:, 0]
                if (Xte is not None and Xte.shape[1] == 1 and reducer is None)
                else None
            )
        )

        # Mean predictions and residuals (normalized) on TRAIN and TEST separately
        Y_pred_train = self._predict_pointwise_mean(est, Xtr, n_samples)  # (ntr, y_dim)
        Resid_train = ytr - Y_pred_train

        if Xte is not None and yte is not None:
            Y_pred_test = self._predict_pointwise_mean(
                est, Xte, n_samples
            )  # (nte, y_dim)
            Resid_test = yte - Y_pred_test
        else:
            Y_pred_test = None
            Resid_test = None

        # --- 2D → 2D path: 2 surfaces (grid from TRAIN) + residual panels ---
        if x_dim == 2 and y_dim == 2:
            self._plot_2d2d_with_curves_and_residuals(
                estimator=est,
                Xtr=Xtr,
                Ytr=ytr,
                Xte=Xte,
                Yte=yte,
                n_samples=n_samples,
                title=title,
                loss_history=loss_history,
                y_pred_test=Y_pred_test,
                resid_test=Resid_test,
                y_pred_train=Y_pred_train if show_train_residuals else None,
                resid_train=Resid_train if show_train_residuals else None,
                grid_res=50,
            )
            return

        # --- general path: one figure per output (fit + residual panels) ---
        has_mdn = self._estimator_has_mdn(est)
        is_prob = self._is_probabilistic(est)

        for out_idx in range(y_dim):
            ytr_1d = ytr[:, out_idx : out_idx + 1]
            yte_1d = yte[:, out_idx : out_idx + 1] if yte is not None else None

            y_pred_tr = Y_pred_train[:, out_idx]
            resid_tr = Resid_train[:, out_idx]

            if Y_pred_test is not None:
                y_pred_te = Y_pred_test[:, out_idx]
                resid_te = Resid_test[:, out_idx]
            else:
                y_pred_te = None
                resid_te = None

            fig = make_subplots(
                rows=4,
                cols=1,
                shared_xaxes=False,
                vertical_spacing=0.12,
                subplot_titles=(
                    f"{title} — y{out_idx} (normalized)",
                    "Training / Validation / Test",
                    "Residuals vs Fitted (test)"
                    + (" + train" if show_train_residuals else ""),
                    "Residual distribution (test)"
                    + (" + train" if show_train_residuals else ""),
                ),
            )

            # Row 1: fit curves using TRAIN ONLY; overlay TEST points if present
            if has_mdn:
                try:
                    self._add_mdn_fit_row(
                        fig,
                        row=1,
                        estimator=est,
                        X=Xtr,
                        X_red=Xtr_red,
                        y_raw_1d=ytr_1d,
                        out_idx=out_idx,
                    )
                except Exception:
                    self._add_prob_fit_row(
                        fig,
                        row=1,
                        estimator=est,
                        X_red=Xtr_red,
                        X=Xtr,
                        y_raw_1d=ytr_1d,
                    )
            elif is_prob:
                self._add_prob_fit_row(
                    fig,
                    row=1,
                    estimator=est,
                    X_red=Xtr_red,
                    X=Xtr,
                    y_raw_1d=ytr_1d,
                )
            else:
                self._add_det_fit_row(
                    fig,
                    row=1,
                    estimator=est,
                    X_red=Xtr_red,
                    X=Xtr,
                    y_raw_1d=ytr_1d,
                )

            # Overlay TRAIN/TEST scatter on the fit row
            if Xtr_red is not None:
                fig.add_trace(
                    go.Scatter(
                        x=Xtr_red,
                        y=ytr_1d[:, 0],
                        mode="markers",
                        name="Train",
                        marker=dict(opacity=0.35),
                    ),
                    row=1,
                    col=1,
                )
                if Xte_red is not None and yte_1d is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=Xte_red,
                            y=yte_1d[:, 0],
                            mode="markers",
                            name="Test",
                            marker=dict(opacity=0.35),
                        ),
                        row=1,
                        col=1,
                    )

            # Row 2: training/validation/test curves
            self._add_loss_curves_row(fig, row=2, loss_history=loss_history)

            # Row 3: residuals vs fitted (TEST, optionally TRAIN overlay)
            if y_pred_te is not None:
                self._add_residuals_vs_fitted(
                    fig,
                    row=3,
                    fitted=y_pred_te,
                    resid=resid_te,
                    output_name=f"y{out_idx} (test)",
                )
            else:
                # no test -> use train so panel isn't empty
                self._add_residuals_vs_fitted(
                    fig,
                    row=3,
                    fitted=y_pred_tr,
                    resid=resid_tr,
                    output_name=f"y{out_idx} (train)",
                )
            if show_train_residuals and y_pred_tr is not None:
                self._add_residuals_vs_fitted(
                    fig,
                    row=3,
                    fitted=y_pred_tr,
                    resid=resid_tr,
                    output_name=f"y{out_idx} (train)",
                )

            # Row 4: residual histogram (TEST, optionally TRAIN overlay)
            if resid_te is not None:
                self._add_residual_hist(
                    fig, row=4, resid=resid_te, output_name=f"y{out_idx} (test)"
                )
            else:
                self._add_residual_hist(
                    fig, row=4, resid=resid_tr, output_name=f"y{out_idx} (train)"
                )
            if show_train_residuals and resid_tr is not None:
                self._add_residual_hist(
                    fig, row=4, resid=resid_tr, output_name=f"y{out_idx} (train)"
                )

            fig.update_xaxes(title_text="X (reduced, normalized)", row=1, col=1)
            fig.update_yaxes(title_text="y (normalized)", row=1, col=1)
            fig.update_layout(template="plotly_white", height=1100)
            fig.show()

    # ---------------------- 2D→2D (surfaces + panels) --------------------- #

    def _plot_2d2d_with_curves_and_residuals(
        self,
        *,
        estimator,
        Xtr: np.ndarray,  # normalized (train)  -> targets (z) for row-1 overlays
        Ytr: np.ndarray,  # normalized (train)  -> grid domain (x,y) for row-1
        Xte: Optional[np.ndarray],
        Yte: Optional[np.ndarray],
        n_samples: int,
        title: str,
        loss_history: Optional[Dict[str, Any]],
        y_pred_test: Optional[np.ndarray],
        resid_test: Optional[np.ndarray],
        y_pred_train: Optional[np.ndarray] = None,
        resid_train: Optional[np.ndarray] = None,
        grid_res: int = 50,
    ) -> None:
        """
        2D→2D case: render two 3D surfaces y1(x1,x2) and y2(x1,x2) plus
        training/validation curves and residual diagnostics. Axes across both
        3D scenes are forced to share the same x/y/z ranges and aspect.
        """

        # --- build grid over TRAIN *X* domain (x1,x2) ---
        x1_min, x1_max = float(np.min(Xtr[:, 0])), float(np.max(Xtr[:, 0]))
        x2_min, x2_max = float(np.min(Xtr[:, 1])), float(np.max(Xtr[:, 1]))
        gx1 = np.linspace(x1_min, x1_max, grid_res)
        gx2 = np.linspace(x2_min, x2_max, grid_res)
        GX1, GX2 = np.meshgrid(gx1, gx2, indexing="xy")
        X_grid = np.stack([GX1.ravel(), GX2.ravel()], axis=1)

        # predict Y on the X-grid (model maps x -> y). Returns (n, 2) for y1,y2
        Yg = self._predict_pointwise_mean(estimator, X_grid, n_samples)
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
                "(x1,x2) → y1 (normalized)",
                "(x1,x2) → y2 (normalized)",
                "Training / Validation / Test",
                "Residuals vs Fitted (y1)",
                "Residuals vs Fitted (y2)",
                "Residual distribution (y1)",
                "Residual distribution (y2)",
                "Residual joint distribution (y1 vs y2)",
            ],
            vertical_spacing=0.06,
            horizontal_spacing=0.07,
            row_heights=[0.42, 0.14, 0.18, 0.18, 0.08],
        )

        # -------- row1: surfaces + TRAIN/TEST point clouds (x on axes; y as height) -----
        fig.add_trace(
            go.Surface(
                x=GX1, y=GX2, z=z_y1, opacity=0.45, showscale=False, name="y1(x)"
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Surface(
                x=GX1, y=GX2, z=z_y2, opacity=0.45, showscale=False, name="y2(x)"
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
                name="Train (y1)",
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
                name="Train (y2)",
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
                    name="Test (y1)",
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
                    name="Test (y2)",
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
                output_name="y1 (test)",
            )
            self._add_residuals_vs_fitted(
                fig,
                row=3,
                col=2,
                fitted=y_pred_test[:, 1],
                resid=resid_test[:, 1],
                output_name="y2 (test)",
            )
        if y_pred_train is not None and resid_train is not None:
            self._add_residuals_vs_fitted(
                fig,
                row=3,
                col=1,
                fitted=y_pred_train[:, 0],
                resid=resid_train[:, 0],
                output_name="y1 (train)",
            )
            self._add_residuals_vs_fitted(
                fig,
                row=3,
                col=2,
                fitted=y_pred_train[:, 1],
                resid=resid_train[:, 1],
                output_name="y2 (train)",
            )

        if resid_test is not None:
            self._add_residual_hist(
                fig, row=4, col=1, resid=resid_test[:, 0], output_name="y1 (test)"
            )
            self._add_residual_hist(
                fig, row=4, col=2, resid=resid_test[:, 1], output_name="y2 (test)"
            )
        if resid_train is not None:
            self._add_residual_hist(
                fig, row=4, col=1, resid=resid_train[:, 0], output_name="y1 (train)"
            )
            self._add_residual_hist(
                fig, row=4, col=2, resid=resid_train[:, 1], output_name="y2 (train)"
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
                title="x1 (norm)",
                range=list(x_rng),
                tickmode="array",
                tickvals=_ticks(x_rng),
            ),
            yaxis=dict(
                title="x2 (norm)",
                range=list(y_rng),
                tickmode="array",
                tickvals=_ticks(y_rng),
            ),
            zaxis=dict(
                title="y1 (norm)",
                range=list(z_rng),
                tickmode="array",
                tickvals=_ticks(z_rng),
            ),
            aspectmode="cube",  # equal scale, no distortion
        )
        scene2_axes = dict(
            xaxis=dict(
                title="x1 (norm)",
                range=list(x_rng),
                tickmode="array",
                tickvals=_ticks(x_rng),
            ),
            yaxis=dict(
                title="x2 (norm)",
                range=list(y_rng),
                tickmode="array",
                tickvals=_ticks(y_rng),
            ),
            zaxis=dict(
                title="y2 (norm)",
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
            height=1600,  # less tall; adjust if you want more vertical space
            width=1600,
            margin=dict(l=40, r=40, t=80, b=40),
        )
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
        loss_history: Optional[Dict[str, Any]],
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
        bins: List[float] = list(loss_history.get("bins", []))
        train_loss: List[Optional[float]] = list(loss_history.get("train_loss", []))
        val_loss: List[Optional[float]] = list(loss_history.get("val_loss", []))
        test_loss: List[Optional[float]] = list(loss_history.get("test_loss", []))
        n_train: List[int] = list(loss_history.get("n_train", []))
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
    ):
        pred = estimator.predict(X, n_samples=max(64, 200))
        arr = np.asarray(pred)
        if arr.ndim == 2:
            arr = arr[:, None, :]  # (n, 1, ydim)
        elif arr.ndim == 3:
            n = X.shape[0]
            if arr.shape[0] == n:
                pass
            elif arr.shape[1] == n:
                arr = np.transpose(arr, (1, 0, 2))
            else:
                arr = np.transpose(arr, (1, 0, 2))
        mean = arr.mean(axis=1)[:, 0]
        p05 = np.percentile(arr, 5, axis=1)[:, 0]
        p95 = np.percentile(arr, 95, axis=1)[:, 0]
        order = np.argsort(X_red)
        fig.add_trace(
            go.Scatter(x=X_red[order], y=mean[order], mode="lines", name="Mean"),
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
        self, reducer, X: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        if X is None:
            return None
        if reducer is None:
            # 1D case handled by caller
            return X[:, 0] if X.shape[1] == 1 else None
        xr = reducer.transform(X).squeeze()
        return xr

    def _predict_pointwise_mean(
        self,
        estimator: Any,
        X: np.ndarray,  # normalized inputs
        n_samples: int,
    ) -> np.ndarray:
        """Return mean prediction in normalized space, shape (n, y_dim)."""
        try:
            y = np.atleast_2d(estimator.predict(X))
            if y.ndim == 2:
                return y
            if y.ndim == 3:
                n = X.shape[0]
                if y.shape[0] == n:
                    return y.mean(axis=1)
                if y.shape[1] == n:
                    return y.mean(axis=0)
        except Exception:
            pass
        if self._estimator_has_mdn(estimator):
            try:
                mp = self._get_mdn_params(estimator, X)
                pi, mu = mp["pi"], mp["mu"]
                if mu.ndim == 2:
                    mu = mu[..., None]
                return np.sum(pi[..., None] * mu, axis=1)
            except Exception:
                pass
        # Prefer analytic/Monte Carlo mean helpers when available
        if hasattr(estimator, "predict_mean"):
            try:
                return np.asarray(
                    estimator.predict_mean(X, n_samples=max(64, n_samples))
                )
            except Exception:
                pass

        if hasattr(estimator, "sample"):
            try:
                s = estimator.sample(X, n_samples=max(64, n_samples))
                s = np.asarray(s)
                if s.ndim == 2:
                    return s
                n = X.shape[0]
                if s.ndim == 3:
                    if s.shape[0] == n:
                        return s.mean(axis=1)
                    if s.shape[1] == n:
                        return s.mean(axis=0)
            except Exception:
                pass

        s = estimator.predict(X)
        s = np.asarray(s)
        if s.ndim == 2:
            return s
        n = X.shape[0]
        if s.ndim == 3:
            if s.shape[0] == n:
                return s.mean(axis=1)
            if s.shape[1] == n:
                return s.mean(axis=0)
        return np.squeeze(s)

    def _is_probabilistic(self, estimator: Any) -> bool:
        try:
            import inspect

            if "n_samples" in inspect.signature(estimator.predict).parameters:
                return True
        except Exception:
            pass
        t = str(getattr(estimator, "type", "")).lower()
        return any(k in t for k in ("mdn", "vae", "prob", "bayes"))

    def _estimator_has_mdn(self, estimator: Any) -> bool:
        return any(
            hasattr(estimator, n)
            for n in ("get_mixture_parameters", "get_mixture_params")
        )

    def _get_mdn_params(self, estimator: Any, X: np.ndarray) -> Dict[str, np.ndarray]:
        params = (
            estimator.get_mixture_parameters(X)
            if hasattr(estimator, "get_mixture_parameters")
            else estimator.get_mixture_params(X)
        )
        if isinstance(params, tuple) and len(params) >= 2:
            pi, mu = params[0], params[1]
            sigma = params[2] if len(params) > 2 else None
            return {
                "pi": np.asarray(pi),
                "mu": np.asarray(mu),
                "sigma": None if sigma is None else np.asarray(sigma),
            }
        if isinstance(params, dict):
            return {
                "pi": np.asarray(params["pi"]),
                "mu": np.asarray(params["mu"]),
                "sigma": None
                if params.get("sigma") is None
                else np.asarray(params["sigma"]),
            }
        raise TypeError("Unsupported MDN parameter format.")
