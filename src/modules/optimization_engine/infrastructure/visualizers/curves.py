from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

from ...domain.analysis.interfaces.base_visualizer import BaseDataVisualizer

try:
    from umap import UMAP

    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False


class ModelCurveVisualizer(BaseDataVisualizer):
    """
    Show-only visualizer of model fit(s) with training/validation/test curve
    embedded in the same figure, plus residual diagnostics.

    Expected `data` keys:
      - estimator
      - X_train, y_train
      - X_test (optional), y_test (optional)
      - X_normalizer, y_normalizer
      - n_samples (int, default 200)
      - non_linear (bool, default False)
      - title (str, optional)
      - loss_history (dict, optional)  # e.g. outcome.loss_history.model_dump()
    """

    # ------------------------------- public ------------------------------- #

    def plot(self, data: Any) -> None:
        if not isinstance(data, dict):
            raise TypeError("ModelCurveVisualizer expects `data` to be a dict.")

        est = data["estimator"]
        Xtr = np.asarray(data["X_train"])
        ytr = np.asarray(data["y_train"])
        Xte = np.asarray(data["X_test"]) if data.get("X_test") is not None else None
        yte = np.asarray(data["y_test"]) if data.get("y_test") is not None else None
        Xn = data["X_normalizer"]
        yn = data["y_normalizer"]

        n_samples = int(data.get("n_samples", 200))
        non_linear = bool(data.get("non_linear", False))
        title = data.get("title", f"Model fit ({type(est).__name__})")
        loss_history: Optional[Dict[str, Any]] = data.get("loss_history")

        # normalize shapes
        if Xtr.ndim == 1:
            Xtr = Xtr.reshape(-1, 1)
        if ytr.ndim == 1:
            ytr = ytr.reshape(-1, 1)
        if Xte is not None and Xte.ndim == 1:
            Xte = Xte.reshape(-1, 1)
        if yte is not None and yte.ndim == 1:
            yte = yte.reshape(-1, 1)

        # concat train+test for overlays
        if Xte is not None and yte is not None:
            X = np.vstack([Xtr, Xte])
            Y = np.vstack([ytr, yte])
            split = np.array([0] * len(Xtr) + [1] * len(Xte))  # 0=train,1=test
        else:
            X, Y = Xtr, ytr
            split = np.zeros(len(Xtr), dtype=int)

        n, x_dim = X.shape
        _, y_dim = Y.shape

        # precompute: reduced axis & normalized inputs & mean predictions on observed X
        X_red = (
            self._reduce_to_1d(X, non_linear)
            if not (x_dim == 2 and y_dim == 2)
            else None
        )
        X_norm = Xn.transform(X)
        Y_pred_mean = self._predict_pointwise_mean(
            est, X_norm, yn, n_samples
        )  # (n, y_dim)
        Resid = Y - Y_pred_mean  # residuals on *observed* points

        # --- 2D → 2D path: 2 surfaces + curves + residual panels ---
        if x_dim == 2 and y_dim == 2:
            self._plot_2d2d_with_curves_and_residuals(
                estimator=est,
                X_raw=X,
                Y_raw=Y,
                X_normer=Xn,
                y_normer=yn,
                n_samples=n_samples,
                title=title,
                split=split,
                loss_history=loss_history,
                y_pred_obs=Y_pred_mean,
                resid_obs=Resid,
                grid_res=50,
            )
            return

        # --- general path: one figure per output (fit + curves + residual panels) ---
        has_mdn = self._estimator_has_mdn(est)
        is_prob = self._is_probabilistic(est)

        for out_idx in range(y_dim):
            y_true = Y[:, out_idx : out_idx + 1]
            y_pred = Y_pred_mean[:, out_idx]
            resid = Resid[:, out_idx]
            fig = make_subplots(
                rows=4,
                cols=1,
                shared_xaxes=False,
                vertical_spacing=0.12,
                subplot_titles=(
                    f"{title} — y{out_idx}",
                    "Training / Validation / Test",
                    "Residuals vs Fitted",
                    "Residual distribution",
                ),
            )

            # Row 1: fit curves
            if has_mdn:
                try:
                    self._add_mdn_fit_row(
                        fig,
                        row=1,
                        estimator=est,
                        X_raw=X,
                        X_red=X_red,
                        X_norm=X_norm,
                        y_raw_1d=y_true,
                        y_normer=yn,
                        out_idx=out_idx,
                    )
                except Exception:
                    self._add_prob_fit_row(
                        fig,
                        row=1,
                        estimator=est,
                        X_red=X_red,
                        X_norm=X_norm,
                        y_raw_1d=y_true,
                        y_normer=yn,
                        n_samples=n_samples,
                        split=split,
                    )
            elif is_prob:
                self._add_prob_fit_row(
                    fig,
                    row=1,
                    estimator=est,
                    X_red=X_red,
                    X_norm=X_norm,
                    y_raw_1d=y_true,
                    y_normer=yn,
                    n_samples=n_samples,
                    split=split,
                )
            else:
                self._add_det_fit_row(
                    fig,
                    row=1,
                    estimator=est,
                    X_red=X_red,
                    X_norm=X_norm,
                    y_raw_1d=y_true,
                    y_normer=yn,
                    split=split,
                )

            # Row 2: training/validation/test curves
            self._add_loss_curves_row(fig, row=2, loss_history=loss_history)

            # Row 3: residuals vs fitted
            self._add_residuals_vs_fitted(
                fig, row=3, fitted=y_pred, resid=resid, output_name=f"y{out_idx}"
            )

            # Row 4: residual histogram
            self._add_residual_hist(fig, row=4, resid=resid, output_name=f"y{out_idx}")

            fig.update_xaxes(title_text="X (reduced)", row=1, col=1)
            fig.update_yaxes(title_text="y", row=1, col=1)
            fig.update_layout(template="plotly_white", height=1100)
            fig.show()

    # ---------------------- 2D→2D (surfaces + panels) --------------------- #

    def _plot_2d2d_with_curves_and_residuals(
        self,
        *,
        estimator,
        X_raw: np.ndarray,
        Y_raw: np.ndarray,
        X_normer,
        y_normer,
        n_samples: int,
        title: str,
        split: np.ndarray,
        loss_history: Optional[Dict[str, Any]],
        y_pred_obs: np.ndarray,  # (n, 2) predictions on observed X
        resid_obs: np.ndarray,  # (n, 2) residuals on observed X
        grid_res: int = 50,
    ) -> None:
        # prediction surfaces on a grid (mean)
        x1_min, x1_max = np.min(X_raw[:, 0]), np.max(X_raw[:, 0])
        x2_min, x2_max = np.min(X_raw[:, 1]), np.max(X_raw[:, 1])
        gx = np.linspace(x1_min, x1_max, grid_res)
        gy = np.linspace(x2_min, x2_max, grid_res)
        GX, GY = np.meshgrid(gx, gy, indexing="xy")
        X_grid = np.stack([GX.ravel(), GY.ravel()], axis=1)
        Xg_norm = X_normer.transform(X_grid)

        Yg = self._predict_pointwise_mean(estimator, Xg_norm, y_normer, n_samples)
        z1 = Yg[:, 0].reshape(grid_res, grid_res)
        z2 = Yg[:, 1].reshape(grid_res, grid_res)

        # Figure with an extra last row for joint residual distribution
        fig = make_subplots(
            rows=5,
            cols=2,
            specs=[
                [{"type": "surface"}, {"type": "surface"}],  # row1: surfaces
                [{"type": "xy", "colspan": 2}, None],  # row2: curves
                [{"type": "xy"}, {"type": "xy"}],  # row3: resid vs fitted (y1,y2)
                [{"type": "xy"}, {"type": "xy"}],  # row4: resid hist (y1,y2)
                [
                    {"type": "xy", "colspan": 2},
                    None,
                ],  # row5: joint residual dist (y1 vs y2)
            ],
            subplot_titles=[
                "(x1,x2) → y1",
                "(x1,x2) → y2",
                "Training / Validation / Test",
                "Residuals vs Fitted (y1)",
                "Residuals vs Fitted (y2)",
                "Residual distribution (y1)",
                "Residual distribution (y2)",
                "Residual joint distribution (y1 vs y2)",
            ],
            vertical_spacing=0.10,
        )

        # row1: surfaces + point clouds
        fig.add_trace(go.Surface(x=GX, y=GY, z=z1, showscale=False), row=1, col=1)
        fig.add_trace(go.Surface(x=GX, y=GY, z=z2, showscale=False), row=1, col=2)

        s0 = split == 0
        s1 = split == 1
        # y1
        fig.add_trace(
            go.Scatter3d(
                x=X_raw[s0, 0],
                y=X_raw[s0, 1],
                z=Y_raw[s0, 0],
                mode="markers",
                name="Train (y1)",
                marker=dict(size=3, opacity=0.5),
            ),
            row=1,
            col=1,
        )
        if split.any():
            fig.add_trace(
                go.Scatter3d(
                    x=X_raw[s1, 0],
                    y=X_raw[s1, 1],
                    z=Y_raw[s1, 0],
                    mode="markers",
                    name="Test (y1)",
                    marker=dict(size=3, opacity=0.5),
                ),
                row=1,
                col=1,
            )
        # y2
        fig.add_trace(
            go.Scatter3d(
                x=X_raw[s0, 0],
                y=X_raw[s0, 1],
                z=Y_raw[s0, 1],
                mode="markers",
                name="Train (y2)",
                marker=dict(size=3, opacity=0.5),
            ),
            row=1,
            col=2,
        )
        if split.any():
            fig.add_trace(
                go.Scatter3d(
                    x=X_raw[s1, 0],
                    y=X_raw[s1, 1],
                    z=Y_raw[s1, 1],
                    mode="markers",
                    name="Test (y2)",
                    marker=dict(size=3, opacity=0.5),
                ),
                row=1,
                col=2,
            )

        # row2: loss curves
        self._add_loss_curves_row(fig, row=2, loss_history=loss_history, col=1)

        # row3: residuals vs fitted
        self._add_residuals_vs_fitted(
            fig,
            row=3,
            col=1,
            fitted=y_pred_obs[:, 0],
            resid=resid_obs[:, 0],
            output_name="y1",
        )
        self._add_residuals_vs_fitted(
            fig,
            row=3,
            col=2,
            fitted=y_pred_obs[:, 1],
            resid=resid_obs[:, 1],
            output_name="y2",
        )

        # row4: residual histograms
        self._add_residual_hist(
            fig, row=4, col=1, resid=resid_obs[:, 0], output_name="y1"
        )
        self._add_residual_hist(
            fig, row=4, col=2, resid=resid_obs[:, 1], output_name="y2"
        )

        # row5: joint residual distribution (y1 vs y2)
        self._add_joint_residual_distribution(
            fig, row=5, col=1, resid_y1=resid_obs[:, 0], resid_y2=resid_obs[:, 1]
        )

        fig.update_layout(
            title=title + " — fit, curves & residuals",
            template="plotly_white",
            height=1600,
        )
        fig.update_scenes(
            xaxis_title="x1", yaxis_title="x2", zaxis_title="y1", row=1, col=1
        )
        fig.update_scenes(
            xaxis_title="x1", yaxis_title="x2", zaxis_title="y2", row=1, col=2
        )
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
        # points
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
        fig.update_xaxes(title_text="Fitted", row=row, col=col)
        fig.update_yaxes(title_text="Residual", row=row, col=col)

    def _add_residual_hist(
        self,
        fig: go.Figure,
        *,
        row: int,
        resid: np.ndarray,
        output_name: str,
        col: int = 1,
    ) -> None:
        # histogram
        fig.add_trace(
            go.Histogram(
                x=resid,
                nbinsx=40,
                name=f"Resid ({output_name})",
                histnorm="probability density",
                opacity=0.7,
            ),
            row=row,
            col=col,
        )
        # vertical zero line drawn as a trace (avoid add_vline/shapes)
        vals = resid[np.isfinite(resid)]
        if vals.size:
            counts, edges = np.histogram(vals, bins=40, density=True)
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
        fig.update_xaxes(title_text="Residual", row=row, col=col)
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
        # 2D residual density (contours/heatmap) + light scatter + zero lines
        vals_x = resid_y1[np.isfinite(resid_y1)]
        vals_y = resid_y2[np.isfinite(resid_y2)]
        fig.add_trace(
            go.Histogram2dContour(
                x=vals_x,
                y=vals_y,
                ncontours=20,
                contours_coloring="heatmap",
                showscale=True,
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
        # zero axes as traces
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
        fig.update_xaxes(title_text="Residual y1", row=row, col=col)
        fig.update_yaxes(title_text="Residual y2", row=row, col=col)

    # ----------------------- loss curves helper row ----------------------- #

    def _add_loss_curves_row(
        self,
        fig: go.Figure,
        row: int,
        loss_history: Optional[Dict[str, Any]],
        col: int = 1,
    ) -> None:
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
        X_norm,
        y_raw_1d,
        y_normer,
        split,
    ):
        y_pred = self._inverse_rows(y_normer, np.atleast_2d(estimator.predict(X_norm)))
        order = np.argsort(X_red)
        fig.add_trace(
            go.Scatter(
                x=X_red[split == 0],
                y=y_raw_1d[split == 0, 0],
                mode="markers",
                name="Train",
                marker=dict(opacity=0.35),
            ),
            row=row,
            col=1,
        )
        if split.any():
            fig.add_trace(
                go.Scatter(
                    x=X_red[split == 1],
                    y=y_raw_1d[split == 1, 0],
                    mode="markers",
                    name="Test",
                    marker=dict(opacity=0.35),
                ),
                row=row,
                col=1,
            )
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
        X_norm,
        y_raw_1d,
        y_normer,
        n_samples,
        split,
    ):
        samples = estimator.predict(X_norm, n_samples=n_samples)
        samples = np.asarray(samples)
        if samples.ndim == 2:
            samples = samples[:, None, :]
        n, ns, ydim = samples.shape
        flat_inv = self._inverse_rows(y_normer, samples.reshape(-1, ydim)).reshape(
            n, ns, ydim
        )
        mean = flat_inv.mean(axis=1)[:, 0]
        p05 = np.percentile(flat_inv, 5, axis=1)[:, 0]
        p95 = np.percentile(flat_inv, 95, axis=1)[:, 0]
        order = np.argsort(X_red)
        fig.add_trace(
            go.Scatter(
                x=X_red[split == 0],
                y=y_raw_1d[split == 0, 0],
                mode="markers",
                name="Train",
                marker=dict(opacity=0.35),
            ),
            row=row,
            col=1,
        )
        if split.any():
            fig.add_trace(
                go.Scatter(
                    x=X_red[split == 1],
                    y=y_raw_1d[split == 1, 0],
                    mode="markers",
                    name="Test",
                    marker=dict(opacity=0.35),
                ),
                row=row,
                col=1,
            )
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
        X_raw,
        X_red,
        X_norm,
        y_raw_1d,
        y_normer,
        out_idx,
    ):
        mp = self._get_mdn_params(estimator, X_norm)
        pi, mu, sigma = mp["pi"], mp["mu"], mp["sigma"]
        if mu.ndim == 2:
            mu = mu[..., None]
        n_pts, K, _ = mu.shape
        mu_k = mu[:, :, out_idx]
        mu_inv = self._inverse_rows(y_normer, mu_k.reshape(-1, 1)).reshape(n_pts, K)

        sigma_inv = None
        if sigma is not None:
            if sigma.ndim == 2:
                sigma = sigma[..., None]
            sig_k = sigma[:, :, out_idx]
            plus = self._inverse_rows(y_normer, (mu_k + sig_k).reshape(-1, 1)).reshape(
                n_pts, K
            )
            minus = self._inverse_rows(y_normer, (mu_k - sig_k).reshape(-1, 1)).reshape(
                n_pts, K
            )
            sigma_inv = (plus - minus) / 2.0

        order = np.argsort(X_red)
        fig.add_trace(
            go.Scatter(
                x=X_red,
                y=y_raw_1d[:, 0],
                mode="markers",
                name="Ground Truth",
                marker=dict(opacity=0.25),
            ),
            row=row,
            col=1,
        )

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
            comp_mu = mu_inv[:, k][order]
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
            if sigma_inv is not None:
                comp_sig = np.clip(sigma_inv[:, k], 1e-12, None)[order]
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

    def _predict_pointwise_mean(
        self,
        estimator: Any,
        X_norm: np.ndarray,
        y_normer,
        n_samples: int,
    ) -> np.ndarray:
        """Mean prediction on the *observed* X (shape: (n, y_dim))."""
        # try direct predict
        try:
            y_pred_norm = np.atleast_2d(estimator.predict(X_norm))
            return self._inverse_rows(y_normer, y_pred_norm)
        except Exception:
            pass

        # try analytic MDN mean
        if self._estimator_has_mdn(estimator):
            try:
                mp = self._get_mdn_params(estimator, X_norm)
                pi, mu = mp["pi"], mp["mu"]
                if mu.ndim == 2:
                    mu = mu[..., None]
                mean_norm = np.sum(pi[..., None] * mu, axis=1)
                return self._inverse_rows(y_normer, mean_norm)
            except Exception:
                pass

        # sampling fallback
        s = estimator.predict(X_norm, n_samples=max(64, n_samples))
        s = np.asarray(s)
        if s.ndim == 2:
            return self._inverse_rows(y_normer, s)
        n, ns, y_dim = s.shape
        flat = s.reshape(-1, y_dim)
        flat_inv = self._inverse_rows(y_normer, flat)
        return flat_inv.reshape(n, ns, y_dim).mean(axis=1)

    def _reduce_to_1d(self, X: np.ndarray, non_linear: bool) -> np.ndarray:
        if X.shape[1] == 1:
            return X[:, 0]
        if non_linear and _HAS_UMAP:
            return UMAP(n_components=1).fit_transform(X).squeeze()
        return PCA(n_components=1).fit_transform(X).squeeze()

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

    def _get_mdn_params(
        self, estimator: Any, X_norm: np.ndarray
    ) -> Dict[str, np.ndarray]:
        params = (
            estimator.get_mixture_parameters(X_norm)
            if hasattr(estimator, "get_mixture_parameters")
            else estimator.get_mixture_params(X_norm)
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

    def _inverse_rows(self, y_normer, arr2d: np.ndarray) -> np.ndarray:
        try:
            return y_normer.inverse_transform(arr2d)
        except Exception:
            return np.vstack(
                [
                    y_normer.inverse_transform(r.reshape(1, -1)).reshape(-1)
                    for r in arr2d
                ]
            )
