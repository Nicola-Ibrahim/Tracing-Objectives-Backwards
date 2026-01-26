import numpy as np


def compute_z_residuals(
    y_pred: np.ndarray, y_target: np.ndarray, tau: np.ndarray
) -> np.ndarray:
    """
    Standardized Z-Residuals (z_j^{(k)}).
    z_j^{(k)} = (ŷ_j^{(k)} - y*_j) / τ_j

    Args:
        y_pred: Predicted objectives, shape (N_test, K_samples, m_objectives)
        y_target: Target objectives, shape (N_test, m_objectives)
        tau: Scale vector, shape (m_objectives,)
    """
    # Expand y_target to (N_test, 1, m) for broadcasting
    y_target_expanded = y_target[:, np.newaxis, :]
    # Expand tau to (1, 1, m) for broadcasting
    tau_expanded = tau[np.newaxis, np.newaxis, :]

    return (y_pred - y_target_expanded) / tau_expanded


def compute_discrepancy_scores(z_residuals: np.ndarray) -> np.ndarray:
    """
    Discrepancy Score (s^{(k)}).
    Euclidean distance of the standardized residual vector.
    s^{(k)} = sqrt(sum(z_j^2))

    Args:
        z_residuals: Standardized residuals, shape (N_test, K_samples, m_objectives)
    Returns:
        Scores of shape (N_test, K_samples)
    """
    return np.linalg.norm(z_residuals, axis=2)
