from enum import Enum

import numpy as np


class PlausibilityScenario(str, Enum):
    BULLSEYE = "The Bullseye"
    SHOTGUN = "The Shotgun"
    SYSTEMATIC_BIAS = "The Systematic Bias"
    UNDETERMINED = "Undetermined"


def compute_mean_residual_vector(z_residuals: np.ndarray) -> np.ndarray:
    """
    Average z across the K candidates for each target.
    Returns z_bar of shape (N_test, m_objectives)
    """
    return np.mean(z_residuals, axis=1)


def compute_systematic_bias(z_bar: np.ndarray) -> np.ndarray:
    """
    Systematic Bias (b).
    b = ‖z_bar‖₂ / √m

    Measures the "target offset" or average proximity of the candidate cluster center
    to the requested objective target, normalized by the number of objectives (m).

    - Low b: The model is "hitting the target" on average.
    - High b: There is a systematic discrepancy between prediction and request.

    Args:
        z_bar: Mean residual vector, shape (N_test, m_objectives)
    """
    m = z_bar.shape[1]
    return np.linalg.norm(z_bar, axis=1) / np.sqrt(m)


def compute_cloud_dispersion(z_residuals: np.ndarray, z_bar: np.ndarray) -> np.ndarray:
    """
    Cloud Dispersion (v).
    v = median_k (‖z^{(k)} - z_bar‖₂ / √m)

    Measures the "within-set spread" or typical distance of individual candidates
    from their own cluster center, normalized by the number of objectives (m).

    - Low v: The candidates are tightly clustered (low variance).
    - High v: The candidates are widely dispersed (high variance).

    Args:
        z_residuals: shape (N_test, K_samples, m_objectives)
        z_bar: shape (N_test, m_objectives)
    """
    m = z_residuals.shape[2]
    # z - z_bar: (N, K, m) - (N, 1, m)
    diff = z_residuals - z_bar[:, np.newaxis, :]
    norms = np.linalg.norm(diff, axis=2) / np.sqrt(m)
    return np.median(norms, axis=1)


def classify_scenario(
    bias: float,
    dispersion: float,
    bias_threshold: float = 0.5,
    dispersion_threshold: float = 0.5,
) -> PlausibilityScenario:
    """
    Categorize the result into accuracy scenarios based on Bias (b) and Dispersion (v).

    Scenarios:
    - **The Bullseye**: Low b, Low v. Accurate and precise predictions.
    - **The Shotgun**: Low b, High v. Accurate on average, but high variance (diffuse).
    - **The Systematic Bias**: High b, Low v. Precise but off-target.
    """
    if bias < bias_threshold and dispersion < dispersion_threshold:
        return PlausibilityScenario.BULLSEYE
    if bias < bias_threshold and dispersion >= dispersion_threshold:
        return PlausibilityScenario.SHOTGUN
    if bias >= bias_threshold and dispersion < dispersion_threshold:
        return PlausibilityScenario.SYSTEMATIC_BIAS

    return PlausibilityScenario.UNDETERMINED
