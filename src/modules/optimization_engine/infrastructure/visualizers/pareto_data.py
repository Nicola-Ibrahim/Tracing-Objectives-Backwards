# plotly_pareto_visualizer.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

from ...domain.visualization.interfaces.base_visualizer import BaseVisualizer


class PlotlyParetoDataVisualizer(BaseVisualizer):
    """
    Pareto dashboard that accepts a dict prepared by the handler.

    Expected keys in `data`:
      - pareto_set            : (n, 2) raw decisions      -> [x1, x2]
      - pareto_front          : (n, 2) raw objectives     -> [f1, f2]
      - pareto_set_norm       : (n, 2) normalized decisions
      - pareto_front_norm     : (n, 2) normalized objectives
      - historical_solutions           : optional (m, 2) raw decisions
      - historical_objectives          : optional (k, 2) raw objectives
      - historical_solutions_norm      : optional (m, 2) normalized decisions
      - historical_objectives_norm     : optional (k, 2) normalized objectives
    """

    _PARETO_COLOR = "#3498db"
    _HISTORY_COLOR = "#D3D3D3"

    # Subplot layout & the keys to read from the handler payload
    _SUBPLOT_CONFIG: Dict[Tuple[int, int], Dict[str, Any]] = {
        # Row 1: RAW 2D scatters
        (1, 1): {
            "type": "scatter_from2d",
            "title": "Decision Space (x1 vs x2)",
            "x_label": "$x_1$",
            "y_label": "$x_2$",
            "array2d_key": "pareto_set",
            "xy_cols": (0, 1),
            "hist_key": "historical_solutions",
            "name": "Pareto Set",
            "symbol": "circle",
            "marker_size": 7,
        },
        (1, 2): {
            "type": "scatter_from2d",
            "title": "Objective Space (f1 vs f2)",
            "x_label": "$f_1$",
            "y_label": "$f_2$",
            "array2d_key": "pareto_front",
            "xy_cols": (0, 1),
            "hist_key": "historical_objectives",
            "name": "Pareto Front",
            "symbol": "circle",
            "marker_size": 7,
        },
        # Row 2: NORMALIZED 2D scatters
        (2, 1): {
            "type": "scatter_from2d",
            "title": "Normalized Decision Space",
            "x_label": "Norm $x_1$",
            "y_label": "Norm $x_2$",
            "array2d_key": "pareto_set_norm",
            "xy_cols": (0, 1),
            "hist_key": "historical_solutions_norm",
            "name": "Norm Pareto Set",
            "symbol": "diamond",
            "marker_size": 6,
        },
        (2, 2): {
            "type": "scatter_from2d",
            "title": "Normalized Objective Space",
            "x_label": "Norm $f_1$",
            "y_label": "Norm $f_2$",
            "array2d_key": "pareto_front_norm",
            "xy_cols": (0, 1),
            "hist_key": "historical_objectives_norm",
            "name": "Norm Pareto Front",
            "symbol": "diamond",
            "marker_size": 6,
        },
        # Row 3/4: 1D PDFs from 2D sources
        (3, 1): {
            "type": "pdf1d_from2d",
            "title": "PDF: x1",
            "source_2d_key": "pareto_set_norm",
            "col": 0,
            "x_label": "Norm $x_1$",
        },
        (3, 2): {
            "type": "pdf1d_from2d",
            "title": "PDF: x2",
            "source_2d_key": "pareto_set_norm",
            "col": 1,
            "x_label": "Norm $x_2$",
        },
        (4, 1): {
            "type": "pdf1d_from2d",
            "title": "PDF: f1",
            "source_2d_key": "pareto_front_norm",
            "col": 0,
            "x_label": "Norm $f_1$",
        },
        (4, 2): {
            "type": "pdf1d_from2d",
            "title": "PDF: f2",
            "source_2d_key": "pareto_front_norm",
            "col": 1,
            "x_label": "Norm $f_2$",
        },
        # Row 5: 2D PDFs from 2D sources
        (5, 1): {
            "type": "pdf2d_from2d",
            "title": "2D PDF: (x1, x2)",
            "source_2d_key": "pareto_set_norm",
            "x_col": 0,
            "y_col": 1,
            "x_label": "Norm $x_1$",
            "y_label": "Norm $x_2$",
        },
        (5, 2): {
            "type": "pdf2d_from2d",
            "title": "2D PDF: (f1, f2)",
            "source_2d_key": "pareto_front_norm",
            "x_col": 0,
            "y_col": 1,
            "x_label": "Norm $f_1$",
            "y_label": "Norm $f_2$",
        },
        # Row 6: 3D from 2D sources (f1, f2, x1/x2) — normalized
        (6, 1): {
            "type": "3d_from2d",
            "title": "3D (normalized): (f1, f2, x1)",
            "x_source_2d_key": "pareto_front_norm",
            "x_col": 0,
            "y_source_2d_key": "pareto_front_norm",
            "y_col": 1,
            "z_source_2d_key": "pareto_set_norm",
            "z_col": 0,
            "x_label": "f1 (norm)",
            "y_label": "f2 (norm)",
            "z_label": "x1 (norm)",
        },
        (6, 2): {
            "type": "3d_from2d",
            "title": "3D (normalized): (f1, f2, x2)",
            "x_source_2d_key": "pareto_front_norm",
            "x_col": 0,
            "y_source_2d_key": "pareto_front_norm",
            "y_col": 1,
            "z_source_2d_key": "pareto_set_norm",
            "z_col": 1,
            "x_label": "f1 (norm)",
            "y_label": "f2 (norm)",
            "z_label": "x2 (norm)",
        },
    }

    _FIGURE_LAYOUT_CONFIG = {
        "title_text": "Pareto Dashboard — Data & PDFs",
        "title_x": 0.5,
        "title_font_size": 24,
        "height": 2300,
        "width": 1600,
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

    def plot(self, data: Dict[str, Any]):
        if not isinstance(data, dict):
            raise TypeError("Visualizer expects a dict prepared by the handler.")

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
                row_heights.append(0.27)
            elif "pdf2d_from2d" in row_types[r]:
                row_heights.append(0.20)
            else:
                row_heights.append(0.16)

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

    def _add_all_subplots(self, fig: go.Figure, data: Dict[str, Any]) -> None:
        for (row, col), cfg in self._SUBPLOT_CONFIG.items():
            t = cfg["type"]

            if t == "scatter_from2d":
                arr2d = self._get_2d(data, cfg["array2d_key"])
                if arr2d.size == 0:
                    continue
                x = arr2d[:, cfg["xy_cols"][0]]
                y = arr2d[:, cfg["xy_cols"][1]]
                self._add_scatter(fig, row, col, x, y, cfg, data)

            elif t == "pdf1d_from2d":
                arr2d = self._get_2d(data, cfg["source_2d_key"])
                if arr2d.size == 0:
                    continue
                v = arr2d[:, cfg["col"]]
                self._add_pdf1d(fig, row, col, v, cfg["x_label"])

            elif t == "pdf2d_from2d":
                arr2d = self._get_2d(data, cfg["source_2d_key"])
                if arr2d.size == 0:
                    continue
                x = arr2d[:, cfg["x_col"]]
                y = arr2d[:, cfg["y_col"]]
                self._add_pdf2d(fig, row, col, x, y, cfg["x_label"], cfg["y_label"])

            elif t == "3d_from2d":
                x_src = self._get_2d(data, cfg["x_source_2d_key"])
                y_src = self._get_2d(data, cfg["y_source_2d_key"])
                z_src = self._get_2d(data, cfg["z_source_2d_key"])
                if min(x_src.size, y_src.size, z_src.size) == 0:
                    continue
                x = x_src[:, cfg["x_col"]]
                y = y_src[:, cfg["y_col"]]
                z = z_src[:, cfg["z_col"]]
                self._add_3d(
                    fig,
                    row,
                    col,
                    x,
                    y,
                    z,
                    cfg["x_label"],
                    cfg["y_label"],
                    cfg["z_label"],
                )

    # ------------------------------ helpers -------------------------------- #

    def _get_2d(self, data: Dict[str, Any], key: str) -> np.ndarray:
        arr = np.asarray(data.get(key, []))
        if arr.size == 0:
            return np.empty((0, 2))
        if arr.ndim == 1:
            # try to coerce to (n, 2) if it was flattened
            n2 = arr.size // 2
            arr = arr.reshape(n2, 2)
        return arr[:, :2]

    def _add_scatter(
        self,
        fig: go.Figure,
        row: int,
        col: int,
        x: np.ndarray,
        y: np.ndarray,
        cfg: Dict[str, Any],
        data: Dict[str, Any],
    ):
        # Main points
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    size=cfg.get("marker_size", 6),
                    opacity=0.8,
                    color=self._PARETO_COLOR,
                    symbol=cfg.get("symbol", "circle"),
                ),
                name=cfg.get("name", "Data"),
                showlegend=True,
                hovertemplate=f"{cfg['x_label']}: %{{x:.4f}}<br>{cfg['y_label']}: %{{y:.4f}}<extra></extra>",
            ),
            row=row,
            col=col,
        )

        # Optional historical overlay (already aligned by handler)
        hist_key = cfg.get("hist_key")
        hist = np.asarray(data.get(hist_key, [])) if hist_key else None
        if hist is not None and hist.size:
            hist = self._get_2d(data, hist_key)
            if hist.size:
                fig.add_trace(
                    go.Scatter(
                        x=hist[:, 0],
                        y=hist[:, 1],
                        mode="markers",
                        marker=dict(
                            size=4,
                            opacity=0.45,
                            color=self._HISTORY_COLOR,
                            symbol="cross",
                        ),
                        name="Historical",
                        showlegend=True,
                    ),
                    row=row,
                    col=col,
                )

        # Axes
        self._set_xy_limits(fig, row, col, x, y)
        fig.update_xaxes(title_text=cfg["x_label"], row=row, col=col)
        fig.update_yaxes(title_text=cfg["y_label"], row=row, col=col)

    def _add_pdf1d(self, fig, row, col, v: np.ndarray, label: str):
        v = v[np.isfinite(v)]
        if v.size == 0:
            return
        # Hist
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
        # KDE
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
        x, y = x[np.isfinite(x)], y[np.isfinite(y)]
        n = min(len(x), len(y))
        if n == 0:
            return
        x, y = x[:n], y[:n]
        try:
            kde = gaussian_kde(np.vstack([x, y]))
            x_min, x_max = float(x.min()), float(x.max())
            y_min, y_max = float(y.min()), float(y.max())
            if x_min == x_max:
                x_min, x_max = x_min - 0.5, x_max + 0.5
            if y_min == y_max:
                y_min, y_max = y_min - 0.5, y_max + 0.5
            gx = np.linspace(x_min, x_max, 120)
            gy = np.linspace(y_min, y_max, 120)
            XX, YY = np.meshgrid(gx, gy)
            ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
            fig.add_trace(
                go.Heatmap(
                    x=gx,
                    y=gy,
                    z=ZZ,
                    colorscale="Viridis",
                    name="Density",
                    showscale=False,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Contour(
                    x=gx,
                    y=gy,
                    z=ZZ,
                    contours=dict(showlines=False),
                    showscale=False,
                    name="Contours",
                ),
                row=row,
                col=col,
            )
        except Exception:
            fig.add_trace(
                go.Histogram2d(
                    x=x,
                    y=y,
                    nbinsx=40,
                    nbinsy=40,
                    colorscale="Viridis",
                    histnorm="probability density",
                    name="2D Histogram",
                ),
                row=row,
                col=col,
            )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(size=3, opacity=0.25, color="black"),
                name="Points",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text=x_label, row=row, col=col)
        fig.update_yaxes(title_text=y_label, row=row, col=col)

    def _add_3d(self, fig, row, col, x, y, z, x_label, y_label, z_label):
        # Keep rows aligned (joint finite mask) — avoids shape distortion
        n = min(len(x), len(y), len(z))
        x, y, z = x[:n], y[:n], z[:n]
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x, y, z = x[m], y[m], z[m]

        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                name="Pareto Points 3D",
                marker=dict(size=3, opacity=0.75, color=self._PARETO_COLOR),
            ),
            row=row,
            col=col,
        )
        fig.update_scenes(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label,
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
