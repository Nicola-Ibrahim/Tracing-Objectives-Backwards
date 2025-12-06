from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ....domain.visualization.interfaces.base_visualizer import BaseVisualizer
from .panels.pdf_1d import add_pdf1d
from .panels.pdf_2d import add_pdf2d
from .panels.scatter_2d import add_scatter_base, add_scatter_overlay, set_xy_limits
from .panels.scatter_3d import add_3d_overlay


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

    # --- Color Palette ---
    # Objectives (Blue Theme)
    _OBJ_TRAIN = "#2980b9"  # Belize Hole
    _OBJ_TEST = "#5dade2"  # Light Blue
    _OBJ_COLORSCALE = "Blues"

    # Decisions (Orange Theme)
    _DEC_TRAIN = "#d35400"  # Pumpkin
    _DEC_TEST = "#e59866"  # Light Orange
    _DEC_COLORSCALE = "Oranges"

    _HISTORY_COLOR = "#bdc3c7"  # Silver
    _DEFAULT_COLOR = "#888888"  # Grey fallback

    # KDE styling
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
            "base_color": _DEC_TRAIN,
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
            "base_color": _OBJ_TRAIN,
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
        # Row 2: Raw 1D KDEs (Decisions - Orange)
        (2, 1): {
            "type": "pdf1d_from2d",
            "title": "Raw KDE: x1 (Dec)",
            "source_2d_key": "pareto_set",
            "col": 0,
            "x_label": "$x_1$",
            "color": _DEC_TRAIN,
        },
        (2, 2): {
            "type": "pdf1d_from2d",
            "title": "Raw KDE: x2 (Dec)",
            "source_2d_key": "pareto_set",
            "col": 1,
            "x_label": "$x_2$",
            "color": _DEC_TRAIN,
        },
        # Row 3: Raw 1D KDEs (Objectives - Blue)
        (3, 1): {
            "type": "pdf1d_from2d",
            "title": "Raw KDE: y1 (Obj)",
            "source_2d_key": "pareto_front",
            "col": 0,
            "x_label": "$y_1$",
            "color": _OBJ_TRAIN,
        },
        (3, 2): {
            "type": "pdf1d_from2d",
            "title": "Raw KDE: y2 (Obj)",
            "source_2d_key": "pareto_front",
            "col": 1,
            "x_label": "$y_2$",
            "color": _OBJ_TRAIN,
        },
        # Row 4: NORMALIZED 2D scatters (train/test only)
        (4, 1): {
            "type": "scatter_many",
            "title": "Normalized Objectives (y1, y2)",
            "x_label": "Norm $y_1$",
            "y_label": "Norm $y_2$",
            "series": [
                {
                    "key": "X_train",
                    "name": "Train (Objectives)",
                    "symbol": "circle-open",
                    "size": 6,
                    "opacity": 0.7,
                    "color": _OBJ_TRAIN,
                },
                {
                    "key": "X_test",
                    "name": "Test (Objectives)",
                    "symbol": "x",
                    "size": 7,
                    "opacity": 0.9,
                    "color": _OBJ_TEST,
                },
            ],
        },
        (4, 2): {
            "type": "scatter_many",
            "title": "Normalized Decisions (x1, x2)",
            "x_label": "Norm $x_1$",
            "y_label": "Norm $x_2$",
            "series": [
                {
                    "key": "y_train",
                    "name": "Train (Decisions)",
                    "symbol": "circle-open",
                    "size": 6,
                    "opacity": 0.7,
                    "color": _DEC_TRAIN,
                },
                {
                    "key": "y_test",
                    "name": "Test (Decisions)",
                    "symbol": "x",
                    "size": 7,
                    "opacity": 0.9,
                    "color": _DEC_TEST,
                },
            ],
        },
        # Row 5/6: PDFs (Normalized)
        (5, 1): {
            "type": "pdf1d_concat",
            "title": "PDF: Norm $y_1$ (Obj)",
            "vec_keys": ["X_train", "X_test"],
            "col": 0,
            "colors": [_OBJ_TRAIN, _OBJ_TEST],
        },
        (5, 2): {
            "type": "pdf1d_concat",
            "title": "PDF: Norm $x_1$ (Dec)",
            "vec_keys": ["y_train", "y_test"],
            "col": 0,
            "colors": [_DEC_TRAIN, _DEC_TEST],
        },
        (6, 1): {
            "type": "pdf1d_concat",
            "title": "PDF: Norm $y_2$ (Obj)",
            "vec_keys": ["X_train", "X_test"],
            "col": 1,
            "colors": [_OBJ_TRAIN, _OBJ_TEST],
        },
        (6, 2): {
            "type": "pdf1d_concat",
            "title": "PDF: Norm $x_2$ (Dec)",
            "vec_keys": ["y_train", "y_test"],
            "col": 1,
            "colors": [_DEC_TRAIN, _DEC_TEST],
        },
        # Row 7: Raw 2D PDFs (Moved from Row 10)
        (7, 1): {
            "type": "pdf2d_concat",
            "title": "2D PDF: Raw Decisions (x1, x2)",
            "mat_keys": ["historical_solutions"],
            "x_col": 0,
            "y_col": 1,
            "x_label": "$x_1$ (Raw)",
            "y_label": "$x_2$ (Raw)",
            "colorscale": _DEC_COLORSCALE,
        },
        (7, 2): {
            "type": "pdf2d_concat",
            "title": "2D PDF: Raw Objectives (y1, y2)",
            "mat_keys": ["historical_objectives"],
            "x_col": 0,
            "y_col": 1,
            "x_label": "$y_1$ (Raw)",
            "y_label": "$y_2$ (Raw)",
            "colorscale": _OBJ_COLORSCALE,
        },
        # Row 8: Normalized 2D PDFs (Shifted from Row 7)
        (8, 1): {
            "type": "pdf2d_concat",
            "title": "2D PDF: (Norm y1, Norm y2) - Obj",
            "mat_keys": ["X_train", "X_test"],
            "x_col": 0,
            "y_col": 1,
            "x_label": "Norm $y_1$",
            "y_label": "Norm $y_2$",
            "colorscale": _OBJ_COLORSCALE,
        },
        (8, 2): {
            "type": "pdf2d_concat",
            "title": "2D PDF: (Norm x1, Norm x2) - Dec",
            "mat_keys": ["y_train", "y_test"],
            "x_col": 0,
            "y_col": 1,
            "x_label": "Norm $x_1$",
            "y_label": "Norm $x_2$",
            "colorscale": _DEC_COLORSCALE,
        },
        # Row 9: 3D (normalized): (x1, x2, y1) and (x1, x2, y2) with train/test overlays
        (9, 1): {
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
                    "color": _DEC_TRAIN,
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
                    "color": _DEC_TEST,
                },
            ],
        },
        (9, 2): {
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
                    "color": _DEC_TRAIN,
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
                    "color": _DEC_TEST,
                },
            ],
        },
        # Row 10: 3D (normalized): (y1, y2, x1) and (y1, y2, x2) Inverse Mapping
        (10, 1): {
            "type": "3d_many",
            "title": "3D (normalized): (y1, y2, x1)",
            "x_label": "y1 (norm)",
            "y_label": "y2 (norm)",
            "z_label": "x1 (norm)",
            "series": [
                {
                    "x_key": "y_train",  # Objectives (y1)
                    "x_col": 0,
                    "y_key": "y_train",  # Objectives (y2)
                    "y_col": 1,
                    "z_key": "X_train",  # Decisions (x1)
                    "z_col": 0,
                    "name": "Train",
                    "size": 3,
                    "opacity": 0.75,
                    "color": _OBJ_TRAIN,
                },
                {
                    "x_key": "y_test",
                    "x_col": 0,
                    "y_key": "y_test",
                    "y_col": 1,
                    "z_key": "X_test",
                    "z_col": 0,
                    "name": "Test",
                    "size": 3,
                    "opacity": 0.85,
                    "color": _OBJ_TEST,
                },
            ],
        },
        (10, 2): {
            "type": "3d_many",
            "title": "3D (normalized): (y1, y2, x2)",
            "x_label": "y1 (norm)",
            "y_label": "y2 (norm)",
            "z_label": "x2 (norm)",
            "series": [
                {
                    "x_key": "y_train",
                    "x_col": 0,
                    "y_key": "y_train",
                    "y_col": 1,
                    "z_key": "X_train",
                    "z_col": 1,
                    "name": "Train",
                    "size": 3,
                    "opacity": 0.75,
                    "color": _OBJ_TRAIN,
                },
                {
                    "x_key": "y_test",
                    "x_col": 0,
                    "y_key": "y_test",
                    "y_col": 1,
                    "z_key": "X_test",
                    "z_col": 1,
                    "name": "Test",
                    "size": 3,
                    "opacity": 0.85,
                    "color": _OBJ_TEST,
                },
            ],
        },
    }

    _FIGURE_LAYOUT_CONFIG = {
        "title_text": "Pareto Dashboard — Raw & Normalized Data",
        "title_x": 0.5,
        "title_font_size": 24,
        "height": 5500,  # Increased height significantly
        "width": 1800,
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
                    add_scatter_base(
                        fig,
                        row,
                        col,
                        x,
                        y,
                        name=cfg.get("base_name", "Data"),
                        color=cfg.get("base_color", self._DEFAULT_COLOR),
                        symbol=cfg.get("base_symbol", "circle"),
                        size=cfg.get("base_size", 6),
                        x_label=cfg["x_label"],
                        y_label=cfg["y_label"],
                    )
                # overlays (raw)
                for ov in cfg.get("overlays", []) or []:
                    arr = self._get_2d(data, ov["key"])
                    if arr.size:
                        add_scatter_overlay(
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
                add_pdf1d(
                    fig,
                    row,
                    col,
                    v,
                    cfg["x_label"],
                    color=cfg.get("color", "#888"),
                )

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
                        add_scatter_overlay(
                            fig,
                            row,
                            col,
                            x,
                            y,
                            name=s.get("name", s["key"]),
                            symbol=s.get("symbol", "circle"),
                            size=s.get("size", 6),
                            opacity=s.get("opacity", 0.8),
                            color=s.get("color", self._DEFAULT_COLOR),
                        )
                if any_plotted:
                    # Derive axes limits from all plotted series
                    x_all = np.concatenate(xs)
                    y_all = np.concatenate(ys)
                    set_xy_limits(fig, row, col, x_all, y_all)
                fig.update_xaxes(title_text=cfg["x_label"], row=row, col=col)
                fig.update_yaxes(title_text=cfg["y_label"], row=row, col=col)

            elif t == "pdf1d_concat":
                # If colors provided for each key, overlay them. Else concat.
                keys = cfg["vec_keys"]
                colors = cfg.get("colors", None)

                if colors and len(colors) == len(keys):
                    # Overlay mode
                    for k, color in zip(keys, colors):
                        arr = self._get_2d(data, k)
                        if arr.size and arr.shape[1] > cfg["col"]:
                            v = arr[:, cfg["col"]]
                            add_pdf1d(
                                fig,
                                row,
                                col,
                                v,
                                cfg["title"],
                                color=color,
                            )
                else:
                    # Concat mode (legacy / fallback)
                    v = self._concat_vecs(data, keys, col=cfg["col"])
                    if v.size:
                        add_pdf1d(fig, row, col, v, cfg["title"])

            elif t == "pdf2d_concat":
                M = self._concat_mats(data, cfg["mat_keys"])
                if M.size:
                    x = M[:, cfg["x_col"]]
                    y = M[:, cfg["y_col"]]
                    add_pdf2d(
                        fig,
                        row,
                        col,
                        x,
                        y,
                        cfg["x_label"],
                        cfg["y_label"],
                        colorscale=cfg.get("colorscale", "Blues"),
                        show_points=False,
                    )

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
                    add_3d_overlay(
                        fig,
                        row,
                        col,
                        x,
                        y,
                        z,
                        name=s.get("name", "Series"),
                        size=s.get("size", 3),
                        opacity=s.get("opacity", 0.8),
                        color=s.get("color", self._DEFAULT_COLOR),
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
