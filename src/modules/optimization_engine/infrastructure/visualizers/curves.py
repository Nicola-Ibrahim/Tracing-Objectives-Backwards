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
    embedded in the same figure (as second row).

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

        if x_dim == 2 and y_dim == 2:
            # 2D->2D surfaces + loss curves in same figure (second row)
            self._plot_surfaces_2d_to_2d_with_loss(
                estimator=est,
                X_raw=X,
                Y_raw=Y,
                X_normer=Xn,
                y_normer=yn,
                n_samples=n_samples,
                title=title,
                show_points=True,
                split=split,
                grid_res=50,
                loss_history=loss_history,
            )
            return

        # 1D reduced fits per output + loss curves in the same figure
        X_red = self._reduce_to_1d(X, non_linear)
        X_norm = Xn.transform(X)
        has_mdn = self._estimator_has_mdn(est)
        is_prob = self._is_probabilistic(est)

        for out_idx in range(y_dim):
            y1 = Y[:, out_idx : out_idx + 1]

            # Build 2-row figure now; fill row 1 with the fit, row 2 with loss curves
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=False,
                vertical_spacing=0.18,
                subplot_titles=(
                    f"{title} — y{out_idx}",
                    "Training / Validation / Test",
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
                        y_raw_1d=y1,
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
                        y_raw_1d=y1,
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
                    y_raw_1d=y1,
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
                    y_raw_1d=y1,
                    y_normer=yn,
                    split=split,
                )

            # Row 2: training/validation/test curves (if present)
            self._add_loss_curves_row(fig, row=2, loss_history=loss_history)

            fig.update_xaxes(title_text="X (reduced)", row=1, col=1)
            fig.update_yaxes(title_text="y", row=1, col=1)
            fig.update_layout(template="plotly_white", height=750)
            fig.show()

    # ---------------- 2D→2D with embedded loss curves ---------------- #

    def _plot_surfaces_2d_to_2d_with_loss(
        self,
        estimator,
        X_raw: np.ndarray,
        Y_raw: np.ndarray,
        X_normer,
        y_normer,
        n_samples: int,
        title: str,
        show_points: bool,
        split: np.ndarray | None,
        grid_res: int,
        loss_history: Optional[Dict[str, Any]],
    ) -> None:
        # Grid over X
        x1_min, x1_max = np.min(X_raw[:, 0]), np.max(X_raw[:, 0])
        x2_min, x2_max = np.min(X_raw[:, 1]), np.max(X_raw[:, 1])
        gx = np.linspace(x1_min, x1_max, grid_res)
        gy = np.linspace(x2_min, x2_max, grid_res)
        GX, GY = np.meshgrid(gx, gy, indexing="xy")
        X_grid = np.stack([GX.ravel(), GY.ravel()], axis=1)
        Xg_norm = X_normer.transform(X_grid)

        # Mean prediction on grid
        Yg = None
        try:
            Yg_norm = np.atleast_2d(estimator.predict(Xg_norm))
            Yg = self._inverse_rows(y_normer, Yg_norm)
        except Exception:
            pass

        if Yg is None and self._estimator_has_mdn(estimator):
            try:
                mp = self._get_mdn_params(estimator, Xg_norm)
                pi, mu = mp["pi"], mp["mu"]
                if mu.ndim == 2:
                    mu = mu[..., None]
                mean_norm = np.sum(pi[..., None] * mu, axis=1)
                Yg = self._inverse_rows(y_normer, mean_norm)
            except Exception:
                Yg = None

        if Yg is None:
            samples = estimator.predict(Xg_norm, n_samples=max(64, n_samples))
            samples = np.asarray(samples)
            if samples.ndim == 2:
                Yg = self._inverse_rows(y_normer, samples)
            else:
                n_pts, ns, y_dim = samples.shape
                flat = samples.reshape(-1, y_dim)
                flat_inv = self._inverse_rows(y_normer, flat)
                Yg = flat_inv.reshape(n_pts, ns, y_dim).mean(axis=1)

        z1 = Yg[:, 0].reshape(grid_res, grid_res)
        z2 = Yg[:, 1].reshape(grid_res, grid_res)

        # Build figure: row 1 has 2 surfaces; row 2 (colspan=2) has loss curves
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "surface"}, {"type": "surface"}],
                [{"type": "xy", "colspan": 2}, None],
            ],
            subplot_titles=[
                "(x1,x2) → y1",
                "(x1,x2) → y2",
                "Training / Validation / Test",
            ],
            vertical_spacing=0.12,
        )

        fig.add_trace(go.Surface(x=GX, y=GY, z=z1, showscale=False), row=1, col=1)
        fig.add_trace(go.Surface(x=GX, y=GY, z=z2, showscale=False), row=1, col=2)

        if show_points:
            s0 = (split == 0) if split is not None else slice(None)
            s1 = (split == 1) if split is not None else np.array([], dtype=bool)

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
            if split is not None and split.any():
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
            if split is not None and split.any():
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

        # Row 2: loss curves
        self._add_loss_curves_row(fig, row=2, loss_history=loss_history, col=1)

        fig.update_layout(
            title=title + " — fit & training curves",
            template="plotly_white",
            height=850,
        )
        # Axes titles for surfaces
        fig.update_scenes(
            xaxis_title="x1", yaxis_title="x2", zaxis_title="y1", row=1, col=1
        )
        fig.update_scenes(
            xaxis_title="x1", yaxis_title="x2", zaxis_title="y2", row=1, col=2
        )
        fig.show()

    # ---------------- Loss curves in a subplot row ---------------- #

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

    # ---------------- 1D helpers (fit row) ---------------- #

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

    # ---------------- Utilities ---------------- #

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
