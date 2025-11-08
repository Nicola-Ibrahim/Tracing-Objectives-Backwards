from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

from ....domain.visualization.interfaces.base_visualizer import BaseVisualizer


class PlotlyDatasetVisualizer(BaseVisualizer):
    """
    Visualizes bi-objective data using a dict payload.

    Expected keys in `data`:
      # RAW (not normalized)
      - pareto_front              : (n, 2) raw objectives [x1, x2]
      - pareto_set                : (n, 2) raw decisions [y1, y2]
      - historical_solutions      : (m, 2) raw decisions
      - historical_objectives     : (k, 2) raw objectives

      # NORMALIZED (already normalized in processing)
      - X_train, X_test           : (.., 2) normalized objectives (Pareto front)
      - y_train, y_test           : (.., 2) normalized decisions
    """

    _PARETO_COLOR = "#3498db"
    _HISTORY_COLOR = "#95a5a6"
    _TRAIN_COLOR = "#2ecc71"
    _TEST_COLOR = "#e74c3c"

    # KDE styling
    _KDE2D_COLORSCALE = "Blues"  # same family as your reference image
    _KDE2D_REVERSE = True  # dark = denser
    _KDE2D_LINE_COLOR = "#0b0b0b"
    _POINT_EDGE = "rgba(255,255,255,0.7)"
    _POINT_SIZE = 4

    # Subplot layout & how to read/overlay series
    _SUBPLOT_CONFIG: dict[tuple[int, int], dict[str, Any]] = {
        # Row 1: RAW 2D scatters
        (1, 1): {
            "type": "scatter_from2d",
            "title": "Decision Space (raw x1 vs x2)",
            "x_label": "$x_1$",
            "y_label": "$x_2$",
            "array2d_key": "pareto_set",
            "xy_cols": (0, 1),
            "base_name": "Pareto Set",
            "base_color": _PARETO_COLOR,
            "base_symbol": "circle",
            "base_size": 7,
            "overlays": [
                {
                    "key": "historical_solutions",
                    "name": "Historical (Decisions)",
                    "symbol": "cross",
                    "size": 5,
                    "opacity": 0.5,
                    "color": _HISTORY_COLOR,
                },
            ],
        },
        (1, 2): {
            "type": "scatter_from2d",
            "title": "Objective Space (raw y1 vs y2)",
            "x_label": "$y_1$",
            "y_label": "$y_2$",
            "array2d_key": "pareto_front",
            "xy_cols": (0, 1),
            "base_name": "Pareto Front",
            "base_color": _PARETO_COLOR,
            "base_symbol": "circle",
            "base_size": 7,
            "overlays": [
                {
                    "key": "historical_objectives",
                    "name": "Historical (Objectives)",
                    "symbol": "cross",
                    "size": 5,
                    "opacity": 0.5,
                    "color": _HISTORY_COLOR,
                },
            ],
        },
        # Row 2 & 3: Raw 1D KDEs
        (2, 1): {
            "type": "pdf1d_from2d",
            "title": "Raw KDE: y1",
            "source_2d_key": "pareto_set",
            "col": 0,
            "x_label": "$y_1$",
        },
        (2, 2): {
            "type": "pdf1d_from2d",
            "title": "Raw KDE: x1",
            "source_2d_key": "pareto_front",
            "col": 0,
            "x_label": "$x_1$",
        },
        (3, 1): {
            "type": "pdf1d_from2d",
            "title": "Raw KDE: y2",
            "source_2d_key": "pareto_set",
            "col": 1,
            "x_label": "$y_2$",
        },
        (3, 2): {
            "type": "pdf1d_from2d",
            "title": "Raw KDE: x2",
            "source_2d_key": "pareto_front",
            "col": 1,
            "x_label": "$x_2$",
        },
        # Row 4: NORMALIZED 2D scatters (train/test only)
        (4, 1): {
            "type": "scatter_many",
            "title": "Normalized Objectives (x1, x2)",
            "x_label": "Norm $x_1$",
            "y_label": "Norm $x_2$",
            "series": [
                {
                    "key": "X_train",
                    "name": "Train (Objectives)",
                    "symbol": "circle-open",
                    "size": 6,
                    "opacity": 0.7,
                    "color": _TRAIN_COLOR,
                },
                {
                    "key": "X_test",
                    "name": "Test (Objectives)",
                    "symbol": "x",
                    "size": 7,
                    "opacity": 0.9,
                    "color": _TEST_COLOR,
                },
            ],
        },
        (4, 2): {
            "type": "scatter_many",
            "title": "Normalized Decisions (y1, y2)",
            "x_label": "Norm $y_1$",
            "y_label": "Norm $y_2$",
            "series": [
                {
                    "key": "y_train",
                    "name": "Train (Decisions)",
                    "symbol": "circle-open",
                    "size": 6,
                    "opacity": 0.7,
                    "color": _TRAIN_COLOR,
                },
                {
                    "key": "y_test",
                    "name": "Test (Decisions)",
                    "symbol": "x",
                    "size": 7,
                    "opacity": 0.9,
                    "color": _TEST_COLOR,
                },
            ],
        },
        # Row 5/6: PDFs (from normalized series, concatenated when present)
        (5, 1): {
            "type": "pdf1d_concat",
            "title": "PDF: Norm $y_1$",
            "vec_keys": ["y_train", "y_test"],
            "col": 0,
        },
        (5, 2): {
            "type": "pdf1d_concat",
            "title": "PDF: Norm $x_1$",
            "vec_keys": ["X_train", "X_test"],
            "col": 0,
        },
        (6, 1): {
            "type": "pdf1d_concat",
            "title": "PDF: Norm $y_2$",
            "vec_keys": ["y_train", "y_test"],
            "col": 1,
        },
        (6, 2): {
            "type": "pdf1d_concat",
            "title": "PDF: Norm $x_2$",
            "vec_keys": ["X_train", "X_test"],
            "col": 1,
        },
        # Row 7: 2D PDFs (from normalized series, concatenated when present)
        (7, 1): {
            "type": "pdf2d_concat",
            "title": "2D PDF: (Norm y1, Norm y2)",
            "mat_keys": ["y_train", "y_test"],
            "x_col": 0,
            "y_col": 1,
            "x_label": "Norm $y_1$",
            "y_label": "Norm $y_2$",
        },
        (7, 2): {
            "type": "pdf2d_concat",
            "title": "2D PDF: (Norm x1, Norm x2)",
            "mat_keys": ["X_train", "X_test"],
            "x_col": 0,
            "y_col": 1,
            "x_label": "Norm $x_1$",
            "y_label": "Norm $x_2$",
        },
        # Row 8: 3D (normalized): (x1, x2, y1) and (x1, x2, y2) with train/test overlays
        (8, 1): {
            "type": "3d_many",
            "title": "3D (normalized): (x1, x2, y1)",
            "x_label": "x1 (norm)",
            "y_label": "x2 (norm)",
            "z_label": "y1 (norm)",
            "series": [
                {
                    "x_key": "X_train",
                    "x_col": 0,
                    "y_key": "X_train",
                    "y_col": 1,
                    "z_key": "y_train",
                    "z_col": 0,
                    "name": "Train",
                    "size": 3,
                    "opacity": 0.75,
                    "color": _TRAIN_COLOR,
                },
                {
                    "x_key": "X_test",
                    "x_col": 0,
                    "y_key": "X_test",
                    "y_col": 1,
                    "z_key": "y_test",
                    "z_col": 0,
                    "name": "Test",
                    "size": 3,
                    "opacity": 0.85,
                    "color": _TEST_COLOR,
                },
            ],
        },
        (8, 2): {
            "type": "3d_many",
            "title": "3D (normalized): (x1, x2, y2)",
            "x_label": "x1 (norm)",
            "y_label": "x2 (norm)",
            "z_label": "y2 (norm)",
            "series": [
                {
                    "x_key": "X_train",
                    "x_col": 0,
                    "y_key": "X_train",
                    "y_col": 1,
                    "z_key": "y_train",
                    "z_col": 1,
                    "name": "Train",
                    "size": 3,
                    "opacity": 0.75,
                    "color": _TRAIN_COLOR,
                },
                {
                    "x_key": "X_test",
                    "x_col": 0,
                    "y_key": "X_test",
                    "y_col": 1,
                    "z_key": "y_test",
                    "z_col": 1,
                    "name": "Test",
                    "size": 3,
                    "opacity": 0.85,
                    "color": _TEST_COLOR,
                },
            ],
        },
    }

    _FIGURE_LAYOUT_CONFIG = {
        "title_text": "Pareto Dashboard — Raw & Normalized Data",
        "title_x": 0.5,
        "title_font_size": 24,
        "height": 3400,  # was 2800
        "width": 1800,  # e.g. was 1600
        "template": "plotly_white",
        "legend_orientation": "h",
        "legend_yanchor": "bottom",
        "legend_y": -0.10,
        "legend_xanchor": "center",
        "legend_x": 0.5,
        "legend_title_text": "",
        "legend_title_font_size": 14,
        "margin_t": 90,
        "margin_b": 90,
        "margin_l": 50,
        "margin_r": 50,
        "font_family": "Arial",
        "font_size": 12,
        "hovermode": "closest",
    }

    def __init__(self, save_path: Path | None = None):
        super().__init__(save_path)
        self._save_path = save_path if save_path else False

    # ------------------------------- public ------------------------------- #

    def plot(self, data: dict[str, Any]):
        if not isinstance(data, dict):
            raise TypeError("Visualizer expects a dict prepared by the handler.")

        # Drop any accidental objects (defensive; handler shouldn't pass them)
        for k in list(data.keys()):
            if "normalizer" in k.lower():
                data.pop(k, None)

        fig = self._create_figure_layout()
        self._add_all_subplots(fig, data)
        fig.show()
        if self._save_path:
            self._save_plot(fig)

    # ----------------------------- layout/build ---------------------------- #

    def _save_plot(self, fig: go.Figure) -> None:
        try:
            fig.write_html(str(self._save_path))
            print(f"Plot saved successfully to {self._save_path}")
        except Exception as e:
            print(f"Error saving plot to {self._save_path}: {e}")

    def _create_figure_layout(self) -> go.Figure:
        rows = max(r for r, _ in self._SUBPLOT_CONFIG.keys())
        cols = 2

        specs = [[None for _ in range(cols)] for _ in range(rows)]
        titles = [None] * (rows * cols)
        row_types = {r: set() for r in range(1, rows + 1)}

        def _subplot_type(t: str) -> str:
            return "scene" if t.startswith("3d") else "xy"

        for (r, c), cfg in self._SUBPLOT_CONFIG.items():
            specs[r - 1][c - 1] = {"type": _subplot_type(cfg["type"])}
            titles[(r - 1) * cols + (c - 1)] = cfg["title"]
            row_types[r].add(cfg["type"])

        row_heights = []
        for r in range(1, rows + 1):
            if any(t.startswith("3d") for t in row_types[r]):
                row_heights.append(0.38)  # ↑ was 0.27; make the 3D row taller
            elif "pdf2d" in row_types[r] or "pdf2d_concat" in row_types[r]:
                row_heights.append(0.18)
            else:
                row_heights.append(0.14)

        fig = make_subplots(
            rows=rows,
            cols=cols,
            specs=specs,
            subplot_titles=titles,
            horizontal_spacing=0.05,
            vertical_spacing=0.07,
            column_widths=[0.5, 0.5],
            row_heights=row_heights,
        )

        fig.update_layout(
            title=dict(
                text=self._FIGURE_LAYOUT_CONFIG["title_text"],
                x=self._FIGURE_LAYOUT_CONFIG["title_x"],
                font=dict(size=self._FIGURE_LAYOUT_CONFIG["title_font_size"]),
            ),
            height=self._FIGURE_LAYOUT_CONFIG["height"],
            width=self._FIGURE_LAYOUT_CONFIG["width"],
            template=self._FIGURE_LAYOUT_CONFIG["template"],
            legend=dict(
                orientation=self._FIGURE_LAYOUT_CONFIG["legend_orientation"],
                yanchor=self._FIGURE_LAYOUT_CONFIG["legend_yanchor"],
                y=self._FIGURE_LAYOUT_CONFIG["legend_y"],
                xanchor=self._FIGURE_LAYOUT_CONFIG["legend_xanchor"],
                x=self._FIGURE_LAYOUT_CONFIG["legend_x"],
                title=dict(
                    text=self._FIGURE_LAYOUT_CONFIG["legend_title_text"],
                    font=dict(
                        size=self._FIGURE_LAYOUT_CONFIG["legend_title_font_size"]
                    ),
                ),
            ),
            margin=dict(
                t=self._FIGURE_LAYOUT_CONFIG["margin_t"],
                b=self._FIGURE_LAYOUT_CONFIG["margin_b"],
                l=self._FIGURE_LAYOUT_CONFIG["margin_l"],
                r=self._FIGURE_LAYOUT_CONFIG["margin_r"],
            ),
            font=dict(
                family=self._FIGURE_LAYOUT_CONFIG["font_family"],
                size=self._FIGURE_LAYOUT_CONFIG["font_size"],
            ),
            hovermode=self._FIGURE_LAYOUT_CONFIG["hovermode"],
        )
        return fig

    # ------------------------------ build plots ---------------------------- #

    def _add_all_subplots(self, fig: go.Figure, data: dict[str, Any]) -> None:
        for (row, col), cfg in self._SUBPLOT_CONFIG.items():
            t = cfg["type"]

            if t == "scatter_from2d":
                base = self._get_2d(data, cfg["array2d_key"])
                if base.size:
                    x = base[:, cfg["xy_cols"][0]]
                    y = base[:, cfg["xy_cols"][1]]
                    self._add_scatter_base(
                        fig,
                        row,
                        col,
                        x,
                        y,
                        name=cfg.get("base_name", "Data"),
                        color=cfg.get("base_color", self._PARETO_COLOR),
                        symbol=cfg.get("base_symbol", "circle"),
                        size=cfg.get("base_size", 6),
                        x_label=cfg["x_label"],
                        y_label=cfg["y_label"],
                    )
                # overlays (raw)
                for ov in cfg.get("overlays", []) or []:
                    arr = self._get_2d(data, ov["key"])
                    if arr.size:
                        self._add_scatter_overlay(
                            fig,
                            row,
                            col,
                            arr[:, 0],
                            arr[:, 1],
                            name=ov.get("name", ov["key"]),
                            symbol=ov.get("symbol", "cross"),
                            size=ov.get("size", 5),
                            opacity=ov.get("opacity", 0.6),
                            color=ov.get("color", self._HISTORY_COLOR),
                        )
                # axes updated by base; if no base but overlays exist, still set titles
                fig.update_xaxes(title_text=cfg["x_label"], row=row, col=col)
                fig.update_yaxes(title_text=cfg["y_label"], row=row, col=col)

            elif t == "pdf1d_from2d":
                arr2d = self._get_2d(data, cfg["source_2d_key"])
                if arr2d.size == 0:
                    continue
                v = arr2d[:, cfg["col"]]
                self._add_pdf1d(fig, row, col, v, cfg["x_label"])

            elif t == "scatter_many":
                any_plotted = False
                xs, ys = [], []
                for s in cfg.get("series", []):
                    arr = self._get_2d(data, s["key"])
                    if arr.size:
                        any_plotted = True
                        x, y = arr[:, 0], arr[:, 1]
                        xs.append(x)
                        ys.append(y)
                        self._add_scatter_overlay(
                            fig,
                            row,
                            col,
                            x,
                            y,
                            name=s.get("name", s["key"]),
                            symbol=s.get("symbol", "circle"),
                            size=s.get("size", 6),
                            opacity=s.get("opacity", 0.8),
                            color=s.get("color", self._PARETO_COLOR),
                        )
                if any_plotted:
                    # Derive axes limits from all plotted series
                    x_all = np.concatenate(xs)
                    y_all = np.concatenate(ys)
                    self._set_xy_limits(fig, row, col, x_all, y_all)
                fig.update_xaxes(title_text=cfg["x_label"], row=row, col=col)
                fig.update_yaxes(title_text=cfg["y_label"], row=row, col=col)

            elif t == "pdf1d_concat":
                v = self._concat_vecs(data, cfg["vec_keys"], col=cfg["col"])
                if v.size:
                    self._add_pdf1d(fig, row, col, v, cfg["title"])

            elif t == "pdf2d_concat":
                M = self._concat_mats(data, cfg["mat_keys"])
                if M.size:
                    x = M[:, cfg["x_col"]]
                    y = M[:, cfg["y_col"]]
                    self._add_pdf2d(fig, row, col, x, y, cfg["x_label"], cfg["y_label"])

            elif t == "3d_many":
                plotted = False
                for s in cfg.get("series", []):
                    xsrc = self._get_2d(data, s["x_key"])
                    ysrc = self._get_2d(data, s["y_key"])
                    zsrc = self._get_2d(data, s["z_key"])
                    if min(xsrc.size, ysrc.size, zsrc.size) == 0:
                        continue
                    x = xsrc[:, s["x_col"]]
                    y = ysrc[:, s["y_col"]]
                    z = zsrc[:, s["z_col"]]
                    self._add_3d_overlay(
                        fig,
                        row,
                        col,
                        x,
                        y,
                        z,
                        name=s.get("name", "Series"),
                        size=s.get("size", 3),
                        opacity=s.get("opacity", 0.8),
                        color=s.get("color", self._PARETO_COLOR),
                    )
                    plotted = True
                if plotted:
                    # Titles for scene axes
                    fig.update_scenes(
                        xaxis_title=cfg["x_label"],
                        yaxis_title=cfg["y_label"],
                        zaxis_title=cfg["z_label"],
                        row=row,
                        col=col,
                    )

    # ------------------------------ helpers -------------------------------- #

    def _get_2d(self, data: dict[str, Any], key: str) -> np.ndarray:
        arr = np.asarray(data.get(key, []))
        if arr.size == 0:
            return np.empty((0, 2))
        if arr.ndim == 1:
            n2 = arr.size // 2
            arr = arr.reshape(n2, 2)
        # ensure 2 columns at most
        return arr[:, :2]

    def _concat_mats(self, data: dict[str, Any], keys: list[str]) -> np.ndarray:
        mats = [self._get_2d(data, k) for k in keys]
        mats = [m for m in mats if m.size]
        if not mats:
            return np.empty((0, 2))
        return np.vstack(mats)

    def _concat_vecs(
        self, data: dict[str, Any], keys: list[str], col: int
    ) -> np.ndarray:
        chunks = []
        for k in keys:
            M = self._get_2d(data, k)
            if M.size and M.shape[1] > col:
                chunks.append(M[:, col])
        if not chunks:
            return np.array([])
        return np.concatenate(chunks)

    def _add_scatter_base(
        self,
        fig: go.Figure,
        row: int,
        col: int,
        x: np.ndarray,
        y: np.ndarray,
        *,
        name: str,
        color: str,
        symbol: str,
        size: int,
        x_label: str,
        y_label: str,
    ):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=name,
                marker=dict(size=size, opacity=0.85, color=color, symbol=symbol),
                hovertemplate=f"{x_label}: %{{x:.4f}}<br>{y_label}: %{{y:.4f}}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        self._set_xy_limits(fig, row, col, x, y)

    def _add_scatter_overlay(
        self,
        fig: go.Figure,
        row: int,
        col: int,
        x: np.ndarray,
        y: np.ndarray,
        *,
        name: str,
        symbol: str,
        size: int,
        opacity: float,
        color: str,
    ):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=name,
                marker=dict(size=size, opacity=opacity, color=color, symbol=symbol),
                showlegend=True,
            ),
            row=row,
            col=col,
        )

    def _add_pdf1d(self, fig, row, col, v: np.ndarray, label: str):
        v = v[np.isfinite(v)]
        if v.size == 0:
            return
        fig.add_trace(
            go.Histogram(
                x=v,
                nbinsx=40,
                histnorm="probability density",
                name="Histogram",
                opacity=0.45,
                marker=dict(color="#888"),
            ),
            row=row,
            col=col,
        )
        try:
            kde = gaussian_kde(v)
            lo, hi = float(v.min()), float(v.max())
            if lo == hi:
                lo, hi = lo - 0.5, hi + 0.5
            grid = np.linspace(lo, hi, 300)
            pdf = kde(grid)
            fig.add_trace(
                go.Scatter(x=grid, y=pdf, mode="lines", name="KDE", line=dict(width=3)),
                row=row,
                col=col,
            )
        finally:
            fig.update_xaxes(title_text=label, row=row, col=col)
            fig.update_yaxes(title_text="Density", row=row, col=col)

    def _add_pdf2d(self, fig, row, col, x, y, x_label, y_label):
        # keep only finite & align
        x, y = x[np.isfinite(x)], y[np.isfinite(y)]
        n = min(len(x), len(y))
        if n == 0:
            return
        x, y = x[:n], y[:n]

        # helper: build grid + KDE
        def _kde_on(xv, yv, q_lo=0.01, q_hi=0.99, ngrid=160):
            kde = gaussian_kde(np.vstack([xv, yv]))
            # compact grid: robust bounds by quantiles
            x_lo, x_hi = float(np.quantile(xv, q_lo)), float(np.quantile(xv, q_hi))
            y_lo, y_hi = float(np.quantile(yv, q_lo)), float(np.quantile(yv, q_hi))
            # small padding
            pad_x = 0.03 * max(1e-9, x_hi - x_lo)
            pad_y = 0.03 * max(1e-9, y_hi - y_lo)
            gx = np.linspace(x_lo - pad_x, x_hi + pad_x, ngrid)
            gy = np.linspace(y_lo - pad_y, y_hi + pad_y, ngrid)
            XX, YY = np.meshgrid(gx, gy)
            ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
            # density at points (used to color points)
            z_pts = kde(np.vstack([x, y]))
            return gx, gy, ZZ, z_pts

        # try KDE, retry with tiny jitter if needed; otherwise fallback to hist2d
        try:
            gx, gy, ZZ, z_pts = _kde_on(x, y)
        except Exception:
            try:
                epsx = 1e-6 * (np.std(x) + 1e-12)
                epsy = 1e-6 * (np.std(y) + 1e-12)
                gx, gy, ZZ, z_pts = _kde_on(
                    x + np.random.normal(0, epsx, size=x.shape),
                    y + np.random.normal(0, epsy, size=y.shape),
                )
            except Exception:
                fig.add_trace(
                    go.Histogram2d(
                        x=x,
                        y=y,
                        nbinsx=40,
                        nbinsy=40,
                        colorscale=self._KDE2D_COLORSCALE,
                        reversescale=self._KDE2D_REVERSE,
                        histnorm="probability density",
                        showscale=False,
                        name="2D Histogram",
                    ),
                    row=row,
                    col=col,
                )
                fig.update_xaxes(title_text=x_label, row=row, col=col)
                fig.update_yaxes(title_text=y_label, row=row, col=col)
                return

        # robust z-limits for nice contrast
        flat = ZZ.ravel()
        zmin = float(np.quantile(flat, 0.02))
        zmax = float(np.quantile(flat, 0.98))
        if zmax <= zmin:
            zmin, zmax = float(flat.min()), float(flat.max())
        levels = 9
        step = (zmax - zmin) / max(levels, 1)

        # Filled, labeled contours (blue) — like your reference
        fig.add_trace(
            go.Contour(
                x=gx,
                y=gy,
                z=ZZ,
                zmin=zmin,
                zmax=zmax,
                colorscale=self._KDE2D_COLORSCALE,
                reversescale=self._KDE2D_REVERSE,
                contours=dict(
                    coloring="fill",
                    showlines=True,
                    showlabels=True,
                    start=zmin,
                    end=zmax,
                    size=step,
                    labelfont=dict(size=10, color="#111"),
                ),
                line=dict(color=self._KDE2D_LINE_COLOR, width=1.1),
                colorbar=dict(title="KDE", tickformat=".3f"),
                name="KDE",
                hovertemplate="p(x,y)=%{z:.4f}<extra></extra>",
                showscale=False,
            ),
            row=row,
            col=col,
        )

        # Points colored by their KDE value (adds a “glow” without clutter)
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    size=self._POINT_SIZE,
                    color=z_pts,
                    colorscale=self._KDE2D_COLORSCALE,
                    reversescale=self._KDE2D_REVERSE,
                    opacity=0.9,
                    line=dict(color=self._POINT_EDGE, width=0.5),
                ),
                name="Points",
                showlegend=False,
                hovertemplate=(
                    f"{x_label}: %{{x:.4f}}<br>"
                    f"{y_label}: %{{y:.4f}}<br>"
                    "p: %{{marker.color:.4f}}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

        # Axes titles + equal aspect so the shape isn’t distorted
        fig.update_xaxes(title_text=x_label, row=row, col=col)
        fig.update_yaxes(title_text=y_label, row=row, col=col)
        subplot_idx = (row - 1) * 2 + col
        fig.update_yaxes(scaleanchor=f"x{subplot_idx}", row=row, col=col)

    def _add_3d_overlay(
        self,
        fig: go.Figure,
        row: int,
        col: int,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        *,
        name: str,
        size: int,
        opacity: float,
        color: str,
    ):
        # Align and mask finite rows
        n = min(len(x), len(y), len(z))
        x, y, z = x[:n], y[:n], z[:n]
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x, y, z = x[m], y[m], z[m]

        if x.size == 0:
            return

        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                name=name,
                marker=dict(size=size, opacity=opacity, color=color),
            ),
            row=row,
            col=col,
        )

    def _set_xy_limits(self, fig, row, col, x, y):
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.min(y)), float(np.max(y))
        xr, yr = x_max - x_min, y_max - y_min
        x_pad = xr * 0.1 if xr > 0 else 0.1
        y_pad = yr * 0.1 if yr > 0 else 0.1
        fig.update_xaxes(range=[x_min - x_pad, x_max + x_pad], row=row, col=col)
        fig.update_yaxes(range=[y_min - y_pad, y_max + y_pad], row=row, col=col)
