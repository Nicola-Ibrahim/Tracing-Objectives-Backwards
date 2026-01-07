from unittest.mock import MagicMock

import numpy as np

from modules.optimization_engine.infrastructure.visualization.comparison.panels.calibration_plot import (
    add_calibration_plot,
)
from modules.optimization_engine.infrastructure.visualization.comparison.panels.error_boxplot import (
    add_error_boxplot,
)
from modules.optimization_engine.infrastructure.visualization.comparison.panels.metrics_bar_plot import (
    add_metrics_bar_plots,
)
from modules.optimization_engine.workflows.inverse_model_comparison import (
    InverseModelEvaluator,
)


def test_evaluator_output_structure():
    evaluator = InverseModelEvaluator()

    # Mock dependencies
    inverse_estimator = MagicMock()
    # Mock sample to return a 3D array (n_test, num_samples, x_dim)
    inverse_estimator.sample.return_value = np.random.rand(10, 20, 2)

    forward_estimator = MagicMock()
    # Mock predict to return a 2D array (n_total_samples, y_dim)
    forward_estimator.predict.return_value = np.random.rand(200, 3)

    test_objectives = np.random.rand(10, 3)
    test_decisions = np.random.rand(10, 2)

    normalizer = MagicMock()
    normalizer.inverse_transform.side_effect = lambda x: x
    normalizer.transform.side_effect = lambda x: x

    # Execute evaluation
    results = evaluator.evaluate(
        inverse_estimator=inverse_estimator,
        forward_estimator=forward_estimator,
        test_objectives=test_objectives,
        decision_normalizer=normalizer,
        objective_normalizer=normalizer,
        test_decisions=test_decisions,
        num_samples=20,
    )

    # Assert structure
    assert "metrics" in results
    assert "accuracy" in results["metrics"]
    assert "uncertainty" in results["metrics"]
    assert "calibration" in results["metrics"]

    assert "detailed_results" in results
    assert "residuals" in results["detailed_results"]
    assert "candidates" in results["detailed_results"]
    assert "calibration_curves" in results["detailed_results"]

    print("Evaluator output structure verification: PASSED")
    return results


def test_visualizer_panels(results):
    results_map = {"TestModel": results}
    color_map = {"TestModel": "blue"}
    model_names = ["TestModel"]

    fig = MagicMock()

    # These should not raise exceptions if the data paths are correct
    try:
        add_calibration_plot(fig, 1, 1, results_map, color_map)
        print("add_calibration_plot: PASSED")
    except Exception as e:
        print(f"add_calibration_plot: FAILED with {e}")
        raise

    try:
        add_error_boxplot(fig, 1, 2, results_map, color_map, model_names)
        print("add_error_boxplot: PASSED")
    except Exception as e:
        print(f"add_error_boxplot: FAILED with {e}")
        raise

    try:
        add_metrics_bar_plots(fig, results_map, color_map, model_names)
        print("add_metrics_bar_plots: PASSED")
    except Exception as e:
        print(f"add_metrics_bar_plots: FAILED with {e}")
        raise


if __name__ == "__main__":
    results = test_evaluator_output_structure()
    test_visualizer_panels(results)
