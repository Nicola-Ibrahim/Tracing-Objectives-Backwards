import numpy as np
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
    ) -> dict[str, float]:
        """
        Validates the inverse estimator by sampling candidates for ALL targets simultaneously.
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
        # Note: Your MDN.sample method supports batch inputs (X) and returns (N, samples, out).
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
        pred_obj_norm_3d = pred_obj_norm_flat.reshape(n_test_actual, n_samples_actual, y_dim)

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
        best_shots = np.min(errors, axis=1) # Shape: (n_test,)
        mean_best_shot = np.mean(best_shots)

        # 2. Reliability (Median error per target) -> Mean across targets
        # The reliability is the median error for each target.
        reliabilities = np.median(errors, axis=1) # Shape: (n_test,)
        mean_reliability = np.mean(reliabilities)

        # 3. Diversity (Std Dev of candidates per target) -> Mean across targets
        # std along sample axis (axis=1), then mean over dimensions (axis=2)
        # This gives one diversity score per target.
        # The diversity score is the standard deviation of the candidates for each target.
        diversities = np.std(candidates_3d, axis=1).mean(axis=1) # Shape: (n_test,)
        mean_diversity = np.mean(diversities)

        return {
            "best_shot_error": float(mean_best_shot),
            "reliability_error": float(mean_reliability),
            "diversity_score": float(mean_diversity),
        }