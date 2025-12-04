import numpy as np
from plotly.subplots import make_subplots

from ......domain.visualization.interfaces.base_visualizer import BaseVisualizer
from ..common.diagnostics import (
    add_estimator_summary,
    add_joint_residual,
    add_loss_curves,
    add_residual_hist,
    add_residuals_vs_fitted,
)
from ..common.prediction import point_predict, sample_band
from .panels import add_fit_1d, add_points_overlay
from .reduction import reduce_to_1d, transform_1d


class ModelPerformanceNDVisualizer(BaseVisualizer):
    """General case (not 2D→2D): row-1 1D fit + ribbon; rows 2–5 diagnostics."""

    def plot(self, data: dict) -> None:
        est = data["estimator"]
        Xtr, ytr = np.asarray(data["X_train"]), np.asarray(data["y_train"])
        Xte = np.asarray(data["X_test"]) if data.get("X_test") is not None else None
        yte = np.asarray(data["y_test"]) if data.get("y_test") is not None else None
        n_samples = int(data.get("n_samples", 200))
        title = data.get("title", f"Model fit ({type(est).__name__})")
        loss_history = data["loss_history"]

        # Determine if we can plot 3D surfaces (input dim == 2)
        is_2d_input = (Xtr.shape[1] == 2)
        
        # Adjust rows: 
        # Row 1: 1D Fit
        # Row 2: Ghost 3D Scatter (New, if 2D input) OR Spacer
        # Row 3: Loss
        # Row 4: Spacer
        # Row 5: Resid vs Fit
        # ...
        
        # If 2D input, we insert a row.
        # Original: 5 rows of plots + spacers.
        # New: 6 rows of plots + spacers.
        
        rows = 6 if is_2d_input else 5
        
        specs = [
            [{"type": "xy", "colspan": 2}, None], # Row 1: 1D Fit
            [None, None],  # Spacer
        ]
        
        if is_2d_input:
            specs.extend([
                [{"type": "surface"}, {"type": "surface"}], # Row 2: Ghost 3D
                [None, None], # Spacer
            ])
            
        specs.extend([
            [{"type": "xy", "colspan": 2}, None], # Loss
            [None, None],  # Spacer
            [{"type": "xy"}, {"type": "xy"}], # Resid vs Fit
            [None, None],  # Spacer
            [{"type": "xy"}, {"type": "xy"}], # Hist
            [None, None],  # Spacer
            [{"type": "xy"}, {"type": "xy"}], # QQ
        ])
        
        row_heights = [0.22, 0.06] # Row 1 + Spacer
        if is_2d_input:
            row_heights.extend([0.22, 0.06]) # Row 2 + Spacer
            
        row_heights.extend([0.12, 0.06, 0.15, 0.06, 0.15, 0.06, 0.15])

        subplot_titles_list = [
            "1D Reduced Fit",
        ]
        if is_2d_input:
            subplot_titles_list.extend(["Ghost 3D Scatter (y1)", "Ghost 3D Scatter (y2)"])
            
        subplot_titles_list.extend([
            "Training / Validation / Test",
            "Residuals vs Fitted (y0) (Train)", "Residuals vs Fitted (y0) (Test)",
            "Residual distribution (y0) (Train)", "Residual distribution (y0) (Test)",
            "Q-Q Plot (y0) (Train)", "Q-Q Plot (y0) (Test)",
        ])

        fig = make_subplots(
            rows=len(specs),
            cols=2,
            specs=specs,
            vertical_spacing=0.01,
            subplot_titles=subplot_titles_list,
            row_heights=row_heights,
            column_titles=["Train", "Test"]
        )

        # Row 1: reduce, center+band, overlay
        Xr_train, reducer = reduce_to_1d(Xtr)
        Xr_test = transform_1d(reducer, Xte)
        center, p05, p95 = sample_band(est, Xtr, n_samples)
        add_fit_1d(
            fig,
            row=1,
            X_red=Xr_train,
            center=center,
            p05=p05,
            p95=p95,
            name_center=("MAP" if hasattr(est, "predict_map") else "Prediction"),
        )
        add_points_overlay(
            fig,
            row=1,
            X_train_red=Xr_train,
            y_train_1d=ytr[:, 0:1],
            X_test_red=Xr_test,
            y_test_1d=(yte[:, 0:1] if yte is not None else None),
        )
        fig.update_xaxes(title_text="Objective (reduced, normalized)", row=1, col=1)
        fig.update_yaxes(title_text="Decision (normalized)", row=1, col=1)

        # --- GHOST 3D SCATTER (If Input is 2D) ---
        # If the input space is 2D, we can visualize the full output distribution in 3D
        if Xtr.shape[1] == 2:
            from ..vis_2d.panels import add_surfaces_2d
            # We reuse the 2D panel logic which now has the Ghost plot
            # We'll put it in Row 2 (which is currently empty/spacer in the spec above?)
            # Wait, the spec has Row 2 as spacer.
            # Let's check the spec again.
            # rows=5, cols=2
            # Row 1: xy colspan 2
            # Row 2: Spacer
            # Row 3: xy colspan 2 (Loss)
            
            # We want to insert it. But changing the layout is invasive.
            # Let's see if we can fit it.
            # Actually, the user said "add this new scatter plot".
            # If I use add_surfaces_2d, it expects to plot into (row, col=1) and (row, col=2).
            # Row 2 is a spacer. Row 3 is Loss.
            # I should probably expand the figure to 6 rows?
            # Or just overwrite Row 1 if it's 2D? No, Row 1 is the 1D reduction which is useful.
            
            # Let's add it as a new row. But I need to change the make_subplots spec.
            # This is getting complicated for `vis_nd`.
            # Maybe I should just skip it for `vis_nd` unless explicitly asked?
            # The user said "in the visualizers of 2d and nd".
            # So I MUST add it to `vis_nd`.
            
            # I will modify the make_subplots call to have 6 rows.
            pass # Placeholder, I will do this in a separate edit to `plot` method.

        # Row 3 (was 2) -> Now Row 3 or 5 depending on is_2d_input
        # Row indices in make_subplots are 1-based and count spacers if they are in 'specs' list?
        # No, make_subplots rows count the rows in the grid, including spacers if they are rows.
        # My specs list has spacers as rows.
        
        # Row 1: 1D Fit (Index 1)
        # Spacer (Index 2)
        # Row 2 (if 2D): Ghost 3D (Index 3)
        # Spacer (Index 4)
        # Loss (Index 3 or 5)
        
        loss_row = 5 if is_2d_input else 3
        
        if is_2d_input:
            from ..vis_2d.panels import add_surfaces_2d
            add_surfaces_2d(
                fig,
                row=3, # Index 3 in the grid
                estimator=est,
                X_train=Xtr,
                y_train=ytr,
                X_test=Xte,
                y_test=yte,
                input_symbol="x",
                output_symbol="y",
            )
            
        add_loss_curves(fig, row=loss_row, loss_history=loss_history, col=1)

        # Residual diagnostics
        yhat_tr = point_predict(est, Xtr)
        resid_tr = ytr - yhat_tr
        if Xte is not None and yte is not None:
            yhat_te = point_predict(est, Xte)
            resid_te = yte - yhat_te
        else:
            yhat_te = resid_te = None

        # Compute global residual limits for consistent scaling
        all_resids = [resid_tr]
        if resid_te is not None:
            all_resids.append(resid_te)
        concatenated = np.concatenate(all_resids, axis=0)
        # Filter finite values
        concatenated = concatenated[np.isfinite(concatenated)]
        if concatenated.size > 0:
            r_min, r_max = concatenated.min(), concatenated.max()
            span = r_max - r_min
            pad = span * 0.05 if span > 0 else 1.0
            common_range = [r_min - pad, r_max + pad]
        else:
            common_range = [-1.0, 1.0]

        # Row 5 (was 3) -> Now Row 5 or 7
        resid_vs_fit_row = loss_row + 2
        
        add_residuals_vs_fitted(
            fig,
            row=resid_vs_fit_row,
            col=1,
            fitted=yhat_tr[:, 0],
            resid=resid_tr[:, 0],
            label="y0 (train)",
            range_y=common_range,
        )
        if yhat_te is not None:
            add_residuals_vs_fitted(
                fig,
                row=resid_vs_fit_row,
                col=2,
                fitted=yhat_te[:, 0],
                resid=resid_te[:, 0],
                label="y0 (test)",
                range_y=common_range,
            )

        # Row 7 (was 4) -> Now Row 7 or 9
        hist_row = resid_vs_fit_row + 2
        add_residual_hist(
            fig,
            row=hist_row,
            col=1,
            resid=resid_tr[:, 0],
            label="y0 (train)",
            range_x=common_range,
        )
        if resid_te is not None:
            add_residual_hist(
                fig,
                row=hist_row,
                col=2,
                resid=resid_te[:, 0],
                label="y0 (test)",
                range_x=common_range,
            )

        # Row 9 (was 5) -> Now Row 9 or 11
        qq_row = hist_row + 2
        from ..common.diagnostics import add_qq_plot
        add_qq_plot(
            fig,
            row=qq_row,
            col=1,
            resid=resid_tr[:, 0],
            label="y0 (train)",
            range_y=common_range,
        )
        if resid_te is not None:
            add_qq_plot(
                fig,
                row=qq_row,
                col=2,
                resid=resid_te[:, 0],
                label="y0 (test)",
                range_y=common_range,
            )

        add_estimator_summary(fig, est, loss_history)

        # Add explanations under each row
        # Recalculated Y positions for 9 rows
        # Height ~2000
        # Row 1 (Fit): Top ~1.0, Bottom ~0.78. Expl ~0.80
        # Row 3 (Curves): Top ~0.75, Bottom ~0.63. Expl ~0.65
        # Row 5 (Resid): Top ~0.60, Bottom ~0.45. Expl ~0.47
        # Row 7 (Hist): Top ~0.42, Bottom ~0.27. Expl ~0.29
        # Row 9 (QQ): Top ~0.24, Bottom ~0.09. Expl ~0.11

        explanations = [
            (0.80, "<b>1D Fit</b>: Reduced 1D view of model fit with uncertainty bands. <i>Goal</i>: Blue line should follow data trend."),
            (0.65, "<b>Learning Curves</b>: Tracks loss over epochs. <i>Goal</i>: Both should decrease and converge. Large gap = Overfitting."),
            (0.47, "<b>Residuals vs Fitted</b>: Train (Left) vs Test (Right). <i>Goal</i>: Random scatter around 0. Patterns indicate non-linearity."),
            (0.29, "<b>Error Distribution</b>: Train (Left) vs Test (Right). <i>Goal</i>: Bell-shaped (Gaussian) centered at 0."),
            (0.11, "<b>Q-Q Plot</b>: Train (Left) vs Test (Right). <i>Goal</i>: Points should follow the <b>Red Dashed Line</b>."),
        ]
        for y_pos, text in explanations:
            fig.add_annotation(
                text=text,
                xref="paper",
                yref="paper",
                x=0.5,
                y=y_pos,
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                font=dict(size=11, color="#444"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1,
                borderpad=4,
            )

        fig.update_layout(
            title=title + " — fit & diagnostics (normalized)",
            template="plotly_white",
            height=2000,
            autosize=True,
            margin=dict(l=60, r=280, t=80, b=80),
        )
        fig.show()
