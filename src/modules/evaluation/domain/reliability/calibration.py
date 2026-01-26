import numpy as np


def compute_pit_values(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """
    Probability Integral Transform.
    PIT_{i,d} = 1/K * count(samples <= truth)

    --- Calibration & PIT (Probability Integral Transform) ---
    This measures "Honesty." It doesn't care if the model missed the target;
    it cares if the model *knew* it might miss.
    - PIT Values: For every test point, we see where the truth fell relative
      to our samples.
    - PIT Curve: If calibrated, this should be a diagonal line (Uniform).
    - U-Shape curve: Model is 'Overconfident' (Truth falls outside its predicted range often).
    - Hump-Shape curve: Model is 'Underconfident' (Truth is always in the middle, but model predicted a huge range).

    Args:
        samples: shape (N_test, K_samples, x_dim)
        truth: shape (N_test, x_dim)
    Returns:
        PIT values of shape (N_test * x_dim,) - flattened for curve plotting
    """
    n_test, k_samples, x_dim = samples.shape
    pit_results = []

    for i in range(n_test):
        for d in range(x_dim):
            true_val = truth[i, d]
            model_samples = samples[i, :, d]
            pit = np.mean(model_samples <= true_val)
            pit_results.append(pit)

    return np.array(pit_results)


def compute_crps(samples: np.ndarray, truth: np.ndarray) -> float:
    """
    Continuous Ranked Probability Score using the energy form.
    CRPS = E|X - y| - 0.5 * E|X - X'|

    --- CRPS ---
    Think of this as the "Total Performance" grade.
    It rewards models that are both 'Accurate' (near the truth) and 'Sharp'
    (not just guessing wildly).
    - A lower CRPS is better.
    - Formula used: E|X - y| (Distance to truth) - 0.5 * E|X - X'| (Spread of guesses).

    Args:
        samples: shape (N_test, K_samples, x_dim)
        truth: shape (N_test, x_dim)
    """
    n_test, k_samples, x_dim = samples.shape
    crps_per_dim = []

    for i in range(n_test):
        for d in range(x_dim):
            true_val = truth[i, d]
            model_samples = samples[i, :, d]

            # Part A: Average distance to truth
            mae_to_true = np.mean(np.abs(model_samples - true_val))

            # Part B: Average distance between samples
            # Vectorized calculation for E|X - X'|
            diff_matrix = np.abs(model_samples[:, np.newaxis] - model_samples)
            expected_dist = np.mean(diff_matrix)

            crps_val = mae_to_true - 0.5 * expected_dist
            crps_per_dim.append(crps_val)

    return float(np.mean(crps_per_dim))


def compute_calibration_error(pit_values: np.ndarray) -> float:
    """
    Mean Absolute Calibration Error (MACE).
    How far is the PIT curve from the ideal diagonal?
    """
    # Sort PIT values to create the empirical CDF
    sorted_pit = np.sort(pit_values)
    cdf_y = np.arange(1, len(sorted_pit) + 1) / len(sorted_pit)

    # Gap from diagonal
    return float(np.mean(np.abs(sorted_pit - cdf_y)))
