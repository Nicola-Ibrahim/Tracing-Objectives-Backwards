import numpy as np
import plotly.graph_objects as go

from ..interfaces.base_estimator import BaseEstimator, ProbabilisticEstimator
from ..interfaces.base_normalizer import BaseNormalizer


class InverseModelValidator:
    """
    Domain service for validating inverse models using a forward simulator.
    Optimized with vectorized operations for batch processing.
    """

    def validate(
        self,
        inverse_estimator: BaseEstimator,
        forward_model: BaseEstimator,
        test_objectives: np.ndarray,
        decision_normalizer: BaseNormalizer,
        objective_normalizer: BaseNormalizer,
        num_samples: int = 250,
        random_state: int = 42,
    ) -> dict[str, any]:
        """
        Validates the inverse estimator by sampling candidates for ALL targets simultaneously.
        Returns detailed results including metrics and raw data for plotting.
        """

        # 1. Validate Input
        if not isinstance(inverse_estimator, ProbabilisticEstimator):
            # Fallback or warning logic here if needed
            pass

        n_test, y_dim = test_objectives.shape

        # ---------------------------------------------------------
        # STEP 1: Vectorized Sampling
        # ---------------------------------------------------------
        # We ask the estimator to sample for ALL test targets at once.
        # Expected Output Shape: (n_test, num_samples, x_dim)
        candidates_3d = inverse_estimator.sample(
            test_objectives,
            n_samples=num_samples,
            seed=random_state,
        )

        # Safety Check: If n_samples=1, sample() might return (N, x_dim).
        # We enforce 3D shape (N, 1, x_dim) to keep logic consistent.
        if candidates_3d.ndim == 2:
            candidates_3d = candidates_3d[:, np.newaxis, :]

        n_test_actual, n_samples_actual, x_dim = candidates_3d.shape

        # ---------------------------------------------------------
        # STEP 2: Flatten for Processing
        # ---------------------------------------------------------
        # Normalizers and Forward Models usually expect 2D arrays (Batch, Features)
        # We merge 'n_test' and 'num_samples' into one giant batch.
        # Shape: (n_test * num_samples, x_dim)
        candidates_flat = candidates_3d.reshape(-1, x_dim)

        # ---------------------------------------------------------
        # STEP 3: Batch Simulation (The Speedup!)
        # ---------------------------------------------------------

        # A. Denormalize decisions (Network Space -> Physics Space)
        candidates_orig_flat = decision_normalizer.inverse_transform(candidates_flat)

        # B. Run Simulator / Forward Model
        # This runs ONE big inference instead of N small ones.
        pred_obj_flat = forward_model.predict(candidates_orig_flat)

        # Ensure numpy array
        if not isinstance(pred_obj_flat, np.ndarray):
            pred_obj_flat = np.array(pred_obj_flat)

        # C. Normalize predicted objectives (Physics Space -> Normalized Space)
        pred_obj_norm_flat = objective_normalizer.transform(pred_obj_flat)

        # ---------------------------------------------------------
        # STEP 4: Reshape and Metric Calculation
        # ---------------------------------------------------------

        # Reshape back to (n_test, num_samples, y_dim) so we can group by target
        pred_obj_norm_3d = pred_obj_norm_flat.reshape(
            n_test_actual, n_samples_actual, y_dim
        )

        # Prepare targets for broadcasting: (n_test, 1, y_dim)
        targets_expanded = test_objectives[:, np.newaxis, :]

        # Calculate Euclidean Distance
        # (N, S, Y) - (N, 1, Y) -> (N, S, Y)
        diff = pred_obj_norm_3d - targets_expanded

        # Norm along the feature axis (y_dim)
        # Errors shape: (n_test, num_samples)
        errors = np.linalg.norm(diff, axis=2)

        # --- Metrics Aggregation ---

        # 1. Best Shot (Min error per target) -> Mean across targets
        # The best shot is the candidate that has the lowest error for each target.
        best_shots = np.min(errors, axis=1)  # Shape: (n_test,)
        mean_best_shot = np.mean(best_shots)

        # 2. Reliability (Median error per target) -> Mean across targets
        # The reliability is the median error for each target.
        reliabilities = np.median(errors, axis=1)  # Shape: (n_test,)
        mean_reliability = np.mean(reliabilities)

        # 3. Diversity (Std Dev of candidates per target) -> Mean across targets
        # std along sample axis (axis=1), then mean over dimensions (axis=2)
        # This gives one diversity score per target.
        # The diversity score is the standard deviation of the candidates for each target.
        diversities = np.std(candidates_3d, axis=1).mean(axis=1)  # Shape: (n_test,)
        mean_diversity = np.mean(diversities)

        return {
            "metrics": {
                "best_shot_error": float(mean_best_shot),
                "reliability_error": float(mean_reliability),
                "diversity_score": float(mean_diversity),
            },
            "raw_errors": errors,  # (n_test, num_samples)
            # We will handle calibration separately if we have the data
        }

    def compare_models(
        self,
        results_map: dict[str, dict],  # {model_name: result_dict}
        test_objectives: np.ndarray,
        test_decisions: np.ndarray | None = None,
        inverse_estimators: dict[str, BaseEstimator] | None = None,
    ) -> go.Figure:
        """
        Generates comparison plots for multiple models in a SINGLE figure.
        Row 1: Calibration Curve (Left), Re-simulation Error Boxplot (Right, Spanned)
        Row 2: Best Shot Error, Reliability Error, Diversity Score (Bar Charts)
        """
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=3,
            specs=[
                [{}, {"colspan": 2}, None],
                [{}, {}, {}],
            ],
            subplot_titles=(
                "Calibration Curve (Quantitative)",
                "Re-simulation Error Boxplot",
                "Best Shot Error (Lower is Better)",
                "Reliability Error (Lower is Better)",
                "Diversity Score (Higher is Better)",
            ),
            vertical_spacing=0.15,
        )

        colors = [
            "blue",
            "red",
            "green",
            "orange",
            "purple",
        ]  # Simple color cycle
        model_names = list(results_map.keys())

        # --- 1. Calibration Curve (Row 1, Col 1) ---
        # Ideal Line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Ideal",
                line=dict(color="black", dash="dash"),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        if test_decisions is not None and inverse_estimators is not None:
            for idx, (model_name, estimator) in enumerate(inverse_estimators.items()):
                color = colors[idx % len(colors)]

                # Sample from estimator for PIT
                # shape: (N, Samples, Dim)
                samples = estimator.sample(test_objectives, n_samples=100)
                if samples.ndim == 2:
                    samples = samples[:, np.newaxis, :]

                n_test, _, x_dim = samples.shape

                # Check if x_dim matches test_decisions dim
                if x_dim != test_decisions.shape[1]:
                    # Mismatch dimension, skip calibration for this model or all?
                    # This happens if models predict different decision space sizes? unlikely.
                    pass

                pit_values = []
                for i in range(n_test):
                    for d in range(x_dim):
                        true_val = test_decisions[i, d]
                        model_samples = samples[i, :, d]
                        pit = np.mean(model_samples <= true_val)
                        pit_values.append(pit)

                pit_values = np.sort(pit_values)
                cdf_y = np.arange(1, len(pit_values) + 1) / len(pit_values)

                fig.add_trace(
                    go.Scatter(
                        x=pit_values,
                        y=cdf_y,
                        mode="lines",
                        name=f"{model_name}",
                        line=dict(color=color, width=2),
                        legendgroup=model_name,
                    ),
                    row=1,
                    col=1,
                )

        # Annotations (Over/Under confident)
        fig.add_annotation(
            x=0.8,
            y=0.2,
            text="Overconfident",
            showarrow=False,
            font=dict(size=10, color="gray"),
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=0.2,
            y=0.8,
            text="Underconfident",
            showarrow=False,
            font=dict(size=10, color="gray"),
            row=1,
            col=1,
        )

        # --- 2. Re-simulation Error Boxplot (Row 1, Col 2-3) ---
        for idx, model_name in enumerate(model_names):
            if model_name in results_map:
                res = results_map[model_name]
                raw_errs = res["raw_errors"]  # (n_test, num_samples)
                best_shots = np.min(raw_errs, axis=1)  # (n_test,)

                color = colors[idx % len(colors)]

                fig.add_trace(
                    go.Box(
                        y=best_shots,
                        name=model_name,  # X-axis label
                        boxmean=True,
                        marker_color=color,
                        showlegend=False,
                        legendgroup=model_name,
                    ),
                    row=1,
                    col=2,
                )

        # --- 3. Metrics Bar Charts (Row 2) ---
        # Data preparation
        best_shots_vals = []
        reliability_vals = []
        diversity_vals = []

        for model_name in model_names:
            metrics = results_map[model_name]["metrics"]
            best_shots_vals.append(metrics["best_shot_error"])
            reliability_vals.append(metrics["reliability_error"])
            diversity_vals.append(metrics["diversity_score"])

        # Row 2, Col 1: Best Shot
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=best_shots_vals,
                marker_color=[colors[i % len(colors)] for i in range(len(model_names))],
                showlegend=False,
                name="Best Shot Error",
            ),
            row=2,
            col=1,
        )

        # Row 2, Col 2: Reliability
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=reliability_vals,
                marker_color=[colors[i % len(colors)] for i in range(len(model_names))],
                showlegend=False,
                name="Reliability Error",
            ),
            row=2,
            col=2,
        )

        # Row 2, Col 3: Diversity
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=diversity_vals,
                marker_color=[colors[i % len(colors)] for i in range(len(model_names))],
                showlegend=False,
                name="Diversity Score",
            ),
            row=2,
            col=3,
        )

        # Layout Update
        fig.update_layout(
            title_text="Model Selection Analysis",
            template="plotly_white",
            height=900,
            width=1400,
            # Axis Titles
            xaxis1_title="Predicted Confidence",
            yaxis1_title="Observed Frequency",
            yaxis2_title="Error",
            # Row 2 Y axes
            yaxis3_title="Error",
            yaxis4_title="Error",
            yaxis5_title="Score",
        )

        return fig
