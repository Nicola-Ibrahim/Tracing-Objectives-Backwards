from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import scipy.interpolate as spi
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler

from ...domain.analysis.interfaces.base_visualizer import BaseDataVisualizer


@dataclass
class ParetoData:
    """
    Comprehensive container for all Pareto optimization data.
    Includes original, normalized values, and (optionally) pre-computed interpolations.
    """

    # Core Pareto Data (original values)
    pareto_set: np.ndarray = field(
        default_factory=lambda: np.array([]),
        metadata={
            "description": "Original decision variables (X) for Pareto optimal solutions. Shape (n_samples, n_decision_vars)."
        },
    )
    pareto_front: np.ndarray = field(
        default_factory=lambda: np.array([]),
        metadata={
            "description": "Original objective function values (F) for Pareto optimal solutions. Shape (n_samples, n_objective_vars)."
        },
    )
    historical_solutions: np.ndarray | None = field(
        default_factory=lambda: np.array([]),
        metadata={"description": "All decision variables (X) from all generations."},
    )
    historical_objectives: np.ndarray | None = field(
        default_factory=lambda: np.array([]),
        metadata={
            "description": "All objective function values (F) from all generations."
        },
    )

    # Normalized Values (0-1 range)
    norm_f1: np.ndarray = field(
        default_factory=lambda: np.array([]),
        metadata={"description": "Normalized values of objective function 1."},
    )
    norm_f2: np.ndarray = field(
        default_factory=lambda: np.array([]),
        metadata={"description": "Normalized values of objective function 2."},
    )
    norm_x1: np.ndarray = field(
        default_factory=lambda: np.array([]),
        metadata={"description": "Normalized values of decision variable 1."},
    )
    norm_x2: np.ndarray = field(
        default_factory=lambda: np.array([]),
        metadata={"description": "Normalized values of decision variable 2."},
    )

    # Data for parallel coordinates plot (kept for compatibility; not used here)
    parallel_coordinates_data: np.ndarray = field(
        default_factory=lambda: np.array([]),
        metadata={
            "description": "Combined normalized data for parallel coordinates plot."
        },
    )

    # Interpolations (no longer plotted, kept for compatibility)
    interpolations_1d: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = field(
        default_factory=lambda: {
            "f1_vs_f2": {},
            "f1_vs_x1": {},
            "f1_vs_x2": {},
            "x1_vs_x2": {},
            "f2_vs_x1": {},
            "f2_vs_x2": {},
        }
    )
    interpolations_2d: dict[
        str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    ] = field(
        default_factory=lambda: {
            "f1f2_vs_x1": {},
            "f1f2_vs_x2": {},
        }
    )

    # Scalers used (exposed so we can normalize historical data consistently)
    x1_scaler: MinMaxScaler | None = None
    x2_scaler: MinMaxScaler | None = None
    f1_scaler: MinMaxScaler | None = None
    f2_scaler: MinMaxScaler | None = None


class ParetoDataService:
    """
    Service for preparing Pareto optimization data, including normalization.
    Interpolations are defined but not computed/used in this version.
    """

    INTERPOLATION_METHODS_1D: dict[str, Any] = {
        "Pchip": spi.PchipInterpolator,
        "Cubic Spline": spi.CubicSpline,
        "Linear": "linear",  # for interp1d
        "Quadratic": "quadratic",  # for interp1d
        "RBF": spi.RBFInterpolator,
    }
    INTERPOLATION_METHODS_ND: dict[str, Any] = {
        "Nearest Neighbor": spi.NearestNDInterpolator,
        "Linear ND": spi.LinearNDInterpolator,
    }

    _NUM_INTERPOLATION_POINTS_1D = 100
    _NUM_INTERPOLATION_POINTS_2D_GRID = 50

    def prepare_data(self, raw_pareto_data: Any) -> ParetoData:
        if not hasattr(raw_pareto_data, "pareto_set") or not hasattr(
            raw_pareto_data, "pareto_front"
        ):
            raise ValueError(
                "Raw Pareto data must have 'pareto_set' and 'pareto_front' attributes."
            )
        if (
            raw_pareto_data.pareto_set.size == 0
            or raw_pareto_data.pareto_front.size == 0
        ):
            raise ValueError(
                "Loaded Pareto data (pareto_set or pareto_front) is empty."
            )

        data = ParetoData(
            pareto_set=raw_pareto_data.pareto_set,
            pareto_front=raw_pareto_data.pareto_front,
            historical_solutions=getattr(
                raw_pareto_data, "historical_solutions", np.array([])
            ),
            historical_objectives=getattr(
                raw_pareto_data, "historical_objectives", np.array([])
            ),
        )

        if data.pareto_front.shape[1] < 2:
            raise ValueError("pareto_front must have at least 2 columns for f1 and f2.")
        if data.pareto_set.shape[1] < 2:
            raise ValueError("pareto_set must have at least 2 columns for x1 and x2.")

        # Normalize & store scalers
        data.norm_f1, data.f1_scaler = self._normalize_and_get_scaler(
            data.pareto_front[:, 0]
        )
        data.norm_f2, data.f2_scaler = self._normalize_and_get_scaler(
            data.pareto_front[:, 1]
        )
        data.norm_x1, data.x1_scaler = self._normalize_and_get_scaler(
            data.pareto_set[:, 0]
        )
        data.norm_x2, data.x2_scaler = self._normalize_and_get_scaler(
            data.pareto_set[:, 1]
        )

        data.parallel_coordinates_data = np.hstack(
            [
                data.norm_x1.reshape(-1, 1),
                data.norm_x2.reshape(-1, 1),
                data.norm_f1.reshape(-1, 1),
                data.norm_f2.reshape(-1, 1),
            ]
        )

        # ⚠️ Interpolation computations removed (no longer plotted)

        return data

    def _normalize_and_get_scaler(
        self, data_array: np.ndarray
    ) -> tuple[np.ndarray, MinMaxScaler]:
        if data_array.size == 0:
            return np.array([]), MinMaxScaler()
        scaler = MinMaxScaler()
        arr = data_array.reshape(-1, 1)
        norm = scaler.fit_transform(arr)
        return norm.flatten(), scaler

    def _normalize_with_scaler(
        self, data_array: np.ndarray, scaler: MinMaxScaler
    ) -> np.ndarray:
        if data_array is None or data_array.size == 0 or scaler is None:
            return np.array([])
        return scaler.transform(data_array.reshape(-1, 1)).flatten()


class PlotlyParetoDataVisualizer(BaseDataVisualizer):
    """
    Dashboard for visualizing Pareto set/front:
      • First 4 panels: raw & normalized scatter plots (kept)
      • Added: 1D PDFs for x1, x2, f1, f2; 2D PDFs for (x1,x2) and (f1,f2)
    """

    _PARETO_COLOR = "#3498db"
    _HISTORY_COLOR = "#D3D3D3"

    # --- Only the first 4 scatter plots + new PDF panels ---
    _SUBPLOT_CONFIG: dict[tuple[int, int], dict[str, Any]] = {
        # row 1: raw scatter
        (1, 1): {
            "type": "scatter",
            "title": "Decision Space (x1 vs x2)",
            "x_label": "$x_1$",
            "y_label": "$x_2$",
            "data_key": "pareto_set",
            "data_mapping": {"x": 0, "y": 1},
            "name": "Pareto Set",
            "color": _PARETO_COLOR,
            "symbol": "circle",
            "marker_size": 7,
            "showlegend": True,
            "add_historical_data": "historical_solutions",
        },
        (1, 2): {
            "type": "scatter",
            "title": "Objective Space (f1 vs f2)",
            "x_label": "$f_1$",
            "y_label": "$f_2$",
            "data_key": "pareto_front",
            "data_mapping": {"x": 0, "y": 1},
            "name": "Pareto Front",
            "color": _PARETO_COLOR,
            "symbol": "circle",
            "marker_size": 7,
            "showlegend": True,
            "add_historical_data": "historical_objectives",
        },
        # row 2: normalized scatter
        (2, 1): {
            "type": "scatter",
            "title": "Normalized Decision Space",
            "x_label": "Norm $x_1$",
            "y_label": "Norm $x_2$",
            "x_data_attr": "norm_x1",
            "y_data_attr": "norm_x2",
            "name": "Norm Pareto Set",
            "color": _PARETO_COLOR,
            "symbol": "diamond",
            "marker_size": 6,
            "showlegend": True,
            "add_historical_data": "historical_solutions",
        },
        (2, 2): {
            "type": "scatter",
            "title": "Normalized Objective Space",
            "x_label": "Norm $f_1$",
            "y_label": "Norm $f_2$",
            "x_data_attr": "norm_f1",
            "y_data_attr": "norm_f2",
            "name": "Norm Pareto Front",
            "color": _PARETO_COLOR,
            "symbol": "diamond",
            "marker_size": 6,
            "showlegend": True,
            "add_historical_data": "historical_objectives",
        },
        # row 3: 1D PDFs for x1, x2
        (3, 1): {"type": "pdf1d", "title": "PDF: x1", "var": "x1"},
        (3, 2): {"type": "pdf1d", "title": "PDF: x2", "var": "x2"},
        # row 4: 1D PDFs for f1, f2
        (4, 1): {"type": "pdf1d", "title": "PDF: f1", "var": "f1"},
        (4, 2): {"type": "pdf1d", "title": "PDF: f2", "var": "f2"},
        # row 5: 2D PDFs
        (5, 1): {
            "type": "pdf2d",
            "title": "2D PDF: (x1, x2)",
            "x_var": "x1",
            "y_var": "x2",
        },
        (5, 2): {
            "type": "pdf2d",
            "title": "2D PDF: (f1, f2)",
            "x_var": "f1",
            "y_var": "f2",
        },
        (6, 1): {
            "type": "pdf1d_overlay",
            "title": "Overlaid KDE: Norm x1 & Norm x2",
            "vars": ["x1", "x2"],
        },
        (6, 2): {
            "type": "pdf1d_overlay",
            "title": "Overlaid KDE: Norm f1 & Norm f2",
            "vars": ["f1", "f2"],
        },
    }

    _FIGURE_LAYOUT_CONFIG: dict[str, Any] = {
        "title_text": "Pareto Dashboard — Data & PDFs",
        "title_x": 0.5,
        "title_font_size": 24,
        "height": 2100,
        "width": 1600,
        "showlegend": True,
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
        self._data_service = ParetoDataService()
        self._save_path = save_path if save_path else False

    # ------------------------------- public ------------------------------- #

    def plot(self, data: Any):
        data = self._data_service.prepare_data(data)
        fig = self._create_figure_layout()
        self._add_all_subplots_from_config(fig, data)
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

        def _subplot_type(panel_type: str) -> str:
            # All these are 2D cartesian plots -> "xy"
            if panel_type in ("scatter", "pdf1d", "pdf2d", "pdf1d_overlay"):
                return "xy"

            # otherwise pass through (e.g., "scene", "domain", etc., if you ever add them)
            return panel_type

        for (r, c), cfg in self._SUBPLOT_CONFIG.items():
            specs[r - 1][c - 1] = {"type": _subplot_type(cfg["type"])}
            titles[(r - 1) * cols + (c - 1)] = cfg["title"]

        fig = make_subplots(
            rows=rows,
            cols=cols,
            specs=specs,
            subplot_titles=titles,
            horizontal_spacing=0.05,
            vertical_spacing=0.07,
            column_widths=[0.5, 0.5],
            row_heights=[0.15, 0.15, 0.15, 0.15, 0.20, 0.20],
        )

        fig.update_layout(
            title=dict(
                text=self._FIGURE_LAYOUT_CONFIG["title_text"],
                x=self._FIGURE_LAYOUT_CONFIG["title_x"],
                font=dict(size=self._FIGURE_LAYOUT_CONFIG["title_font_size"]),
            ),
            height=self._FIGURE_LAYOUT_CONFIG["height"],
            width=self._FIGURE_LAYOUT_CONFIG["width"],
            showlegend=self._FIGURE_LAYOUT_CONFIG["showlegend"],
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

    def _add_all_subplots_from_config(self, fig: go.Figure, data: ParetoData) -> None:
        for (row, col), config in self._SUBPLOT_CONFIG.items():
            ptype = config.get("type")

            if ptype == "scatter":
                # historical overlay if present
                historical_attr = config.get("add_historical_data")
                if historical_attr:
                    raw_hist = getattr(data, historical_attr, None)
                    if raw_hist is not None and raw_hist.size > 0:
                        hx, hy = raw_hist[:, 0], raw_hist[:, 1]
                        is_norm = config.get("x_data_attr", "").startswith("norm_")
                        if is_norm:
                            if historical_attr == "historical_solutions":
                                hx = self._data_service._normalize_with_scaler(
                                    hx, data.x1_scaler
                                )
                                hy = self._data_service._normalize_with_scaler(
                                    hy, data.x2_scaler
                                )
                            else:
                                hx = self._data_service._normalize_with_scaler(
                                    hx, data.f1_scaler
                                )
                                hy = self._data_service._normalize_with_scaler(
                                    hy, data.f2_scaler
                                )
                        self._add_scatter_plot(
                            fig,
                            row,
                            col,
                            hx,
                            hy,
                            {
                                "name": "Historical Points",
                                "color": self._HISTORY_COLOR,
                                "symbol": "cross",
                                "marker_size": 4,
                                "showlegend": True,
                                "hovertemplate_name": "Historical",
                                "x_label": config["x_label"],
                                "y_label": config["y_label"],
                            },
                        )

                # main data
                if "data_key" in config:
                    src = getattr(data, config["data_key"], None)
                    x = self._get_data_from_indexed_source(
                        src, config["data_mapping"]["x"]
                    )
                    y = self._get_data_from_indexed_source(
                        src, config["data_mapping"]["y"]
                    )
                else:
                    x = getattr(data, config.get("x_data_attr"), None)
                    y = getattr(data, config.get("y_data_attr"), None)

                if x is None or y is None or x.size == 0 or y.size == 0:
                    print(
                        f"Warning: Insufficient data for {config.get('title','scatter')} at ({row},{col})."
                    )
                    continue

                self._add_scatter_plot(fig, row, col, x, y, config)

            elif ptype == "pdf1d":
                var = config.get("var")
                arr, label = self._get_variable_array_and_label(data, var)
                if arr.size == 0:
                    print(f"Warning: No data for 1D PDF '{var}'.")
                    continue
                self._add_pdf_1d(fig, row, col, arr, label)

            elif ptype == "pdf2d":
                xvar, yvar = config.get("x_var"), config.get("y_var")
                x_arr, x_lab = self._get_variable_array_and_label(data, xvar)
                y_arr, y_lab = self._get_variable_array_and_label(data, yvar)
                if x_arr.size == 0 or y_arr.size == 0:
                    print(f"Warning: No data for 2D PDF '{xvar},{yvar}'.")
                    continue
                self._add_pdf_2d(fig, row, col, x_arr, y_arr, x_lab, y_lab)

            elif ptype == "pdf1d_overlay":
                vlist = config.get("vars", [])
                if not vlist or len(vlist) < 2:
                    print(
                        f"Warning: pdf1d_overlay at ({row},{col}) needs at least two vars."
                    )
                    continue

                series = []
                # choose two colors that contrast well
                colors = [
                    "#1f77b4",
                    "#d62728",
                    "#2ca02c",
                    "#9467bd",
                ]  # in case you add more later
                for i, vname in enumerate(vlist):
                    arr, lab = self._get_variable_array_and_label(data, vname)
                    if arr.size == 0:
                        print(f"Warning: No data for overlay var '{vname}'.")
                        continue
                    series.append((arr, lab, colors[i % len(colors)]))

                if len(series) >= 2:
                    self._add_pdf_1d_overlay(fig, row, col, series)

            else:
                print(f"Warning: Unsupported plot type '{ptype}' at ({row},{col}).")

    # -------------------------------- helpers ------------------------------ #

    def _add_pdf_1d_overlay(
        self,
        fig: go.Figure,
        row: int,
        col: int,
        series: list[tuple[np.ndarray, str, str]],
    ) -> None:
        """
        Overlay multiple 1D KDE curves in the same axes.
        `series` = [(data_array, label, color), ...]
        """
        # build a common X grid spanning all series
        mins, maxs = [], []
        clean_series = []
        for arr, lab, color in series:
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            clean_series.append((arr, lab, color))
            mins.append(float(arr.min()))
            maxs.append(float(arr.max()))

        if not clean_series:
            return

        x_min, x_max = float(min(mins)), float(max(maxs))
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5
        grid = np.linspace(x_min, x_max, 400)

        # plot each KDE
        for arr, lab, color in clean_series:
            try:
                kde = gaussian_kde(arr)
                pdf = kde(grid)
                fig.add_trace(
                    go.Scatter(
                        x=grid,
                        y=pdf,
                        mode="lines",
                        name=f"KDE {lab}",
                        line=dict(width=3, color=color),
                    ),
                    row=row,
                    col=col,
                )
            except Exception:
                # fallback: show a semi-transparent histogram if KDE fails
                fig.add_trace(
                    go.Histogram(
                        x=arr,
                        nbinsx=40,
                        histnorm="probability density",
                        name=f"Hist {lab}",
                        opacity=0.35,
                        marker=dict(color=color),
                    ),
                    row=row,
                    col=col,
                )

        # axes labels: use the first label on x; density on y
        fig.update_xaxes(title_text=clean_series[0][1], row=row, col=col)
        fig.update_yaxes(title_text="Density", row=row, col=col)

    def _get_variable_array_and_label(
        self, data: ParetoData, name: str
    ) -> tuple[np.ndarray, str]:
        if name == "x1":
            return data.norm_x1, "Norm $x_1$"
        if name == "x2":
            return data.norm_x2, "Norm $x_2$"
        if name == "f1":
            return data.norm_f1, "Norm $f_1$"
        if name == "f2":
            return data.norm_f2, "Norm $f_2$"
        return np.array([]), name

    def _get_data_from_indexed_source(
        self, data_source: np.ndarray | tuple, index: int | None
    ) -> np.ndarray | None:
        if data_source is None or index is None:
            return None
        if isinstance(data_source, np.ndarray):
            return (
                data_source[:, index]
                if data_source.ndim > 1
                else (data_source if index == 0 else None)
            )
        elif isinstance(data_source, (list, tuple)):
            try:
                return np.asarray(data_source[index])
            except (IndexError, TypeError):
                return None
        return None

    def _add_scatter_plot(
        self,
        fig: go.Figure,
        row: int,
        col: int,
        x: np.ndarray,
        y: np.ndarray,
        config: dict[str, Any],
    ) -> None:
        name = config.get("name", "Data Points")
        hover_name = config.get("hovertemplate_name", name)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    size=config.get("marker_size", 7),
                    opacity=0.8,
                    color=config.get("color", "#3498db"),
                    symbol=config.get("symbol", "circle"),
                ),
                name=name,
                showlegend=config.get("showlegend", True),
                hovertemplate=f"{config['x_label']}: %{{x:.4f}}<br>{config['y_label']}: %{{y:.4f}}<extra>{hover_name}</extra>",
            ),
            row=row,
            col=col,
        )
        self._set_axis_limits(fig, row, col, x, y)
        fig.update_xaxes(title_text=config["x_label"], row=row, col=col)
        fig.update_yaxes(title_text=config["y_label"], row=row, col=col)

    def _add_pdf_1d(
        self, fig: go.Figure, row: int, col: int, data: np.ndarray, label: str
    ) -> None:
        data = data[np.isfinite(data)]
        if data.size == 0:
            return
        # histogram (density)
        fig.add_trace(
            go.Histogram(
                x=data,
                nbinsx=40,
                histnorm="probability density",
                name="Histogram",
                opacity=0.45,
                marker=dict(color="#888888"),
            ),
            row=row,
            col=col,
        )
        # KDE curve
        try:
            kde = gaussian_kde(data)
            x_min, x_max = float(data.min()), float(data.max())
            if x_min == x_max:
                x_min -= 0.5
                x_max += 0.5
            grid = np.linspace(x_min, x_max, 300)
            pdf = kde(grid)
            fig.add_trace(
                go.Scatter(x=grid, y=pdf, mode="lines", name="KDE", line=dict(width=3)),
                row=row,
                col=col,
            )
            fig.update_xaxes(title_text=label, row=row, col=col)
            fig.update_yaxes(title_text="Density", row=row, col=col)
        except Exception:
            # KDE failed; histogram alone is fine
            fig.update_xaxes(title_text=label, row=row, col=col)
            fig.update_yaxes(title_text="Density", row=row, col=col)

    def _add_pdf_2d(
        self,
        fig: go.Figure,
        row: int,
        col: int,
        x: np.ndarray,
        y: np.ndarray,
        x_label: str,
        y_label: str,
    ) -> None:
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        n = min(len(x), len(y))
        if n == 0:
            return
        x = x[:n]
        y = y[:n]

        # Try KDE 2D
        try:
            kde = gaussian_kde(np.vstack([x, y]))
            x_min, x_max = float(np.min(x)), float(np.max(x))
            y_min, y_max = float(np.min(y)), float(np.max(y))
            if x_min == x_max:
                x_min -= 0.5
                x_max += 0.5
            if y_min == y_max:
                y_min -= 0.5
                y_max += 0.5
            gx = np.linspace(x_min, x_max, 120)
            gy = np.linspace(y_min, y_max, 120)
            XX, YY = np.meshgrid(gx, gy)
            zz = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)

            # heatmap-levels + contours + faint scatter
            fig.add_trace(
                go.Heatmap(
                    x=gx,
                    y=gy,
                    z=zz,
                    colorscale="Viridis",
                    name="Density",
                    showscale=True,
                    hovertemplate=f"{x_label}: %{{x:.4f}}<br>{y_label}: %{{y:.4f}}<br>pdf: %{{z:.3e}}<extra></extra>",
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Contour(
                    x=gx,
                    y=gy,
                    z=zz,
                    contours=dict(showlines=False),
                    line=dict(width=1),
                    showscale=False,
                    name="Contours",
                ),
                row=row,
                col=col,
            )
        except Exception:
            # Fallback: 2D histogram
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

        # Overlay points (light)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="Points",
                marker=dict(size=3, opacity=0.25, color="black"),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        fig.update_xaxes(title_text=x_label, row=row, col=col)
        fig.update_yaxes(title_text=y_label, row=row, col=col)

    def _set_axis_limits(
        self, fig: go.Figure, row: int, col: int, x: np.ndarray, y: np.ndarray
    ) -> None:
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.min(y)), float(np.max(y))
        xr = x_max - x_min
        yr = y_max - y_min
        x_pad = xr * 0.1 if xr > 0 else 0.1
        y_pad = yr * 0.1 if yr > 0 else 0.1
        fig.update_xaxes(range=[x_min - x_pad, x_max + x_pad], row=row, col=col)
        fig.update_yaxes(range=[y_min - y_pad, y_max + y_pad], row=row, col=col)
