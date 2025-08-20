import cocoex
import numpy as np

from ...domain.model_management.interfaces.base_forward_decision_mapper import (
    BaseForwardDecisionMapper,
)


class CocoexFunctionForwardDecisionMapper(BaseForwardDecisionMapper):
    """
    An implementation of a forward mapper using functions from the cocoex library.
    This class handles the initialization of COCO problems internally.
    It does NOT include a fallback mechanism; failure to initialize a COCO problem
    will result in an error.
    """

    def __init__(
        self,
        suite_name: str = "bbob-biobj",
        function_indices: int = 1,
        instance_indices: int = 1,
        dimensions: int = 2,
    ):
        """
        Initializes the CocoexFunctionForwardDecisionMapper with a specific COCO problem.

        Args:
            suite_name (str): The name of the COCO suite (e.g., "bbob", "bbob-biobj").
            function_indices (int): The index of the function within the suite.
            instance_indices (int): The instance index for the problem.
            dimensions (int): The dimensionality of the input X for the problem.

        Raises:
            ValueError: If the COCO problem cannot be initialized or validation fails.
        """
        self.problem = None  # Will hold the cocoex.Problem object

        # Validate function index for bbob-biobj (if applicable)
        if suite_name == "bbob-biobj" and not (1 <= function_indices <= 55):
            raise ValueError(
                f"`function_indices` must be between 1 and 55 for suite '{suite_name}', got {function_indices}."
            )

        suite_options = (
            f"dimensions:{dimensions} "
            f"instance_indices:{instance_indices} "
            f"function_indices:{function_indices}"
        )

        try:
            suite = cocoex.Suite(
                suite_name,  # suite_name
                "",  # suite_instance
                suite_options,  # suite_options (must only use valid keys!)
            )
            self.problem = suite.get_problem(0)

            print(
                f"CocoexFunctionForwardDecisionMapper initialized with: {self.problem.name} "
                f"(dimension: {self.problem.dimension}, instance: {self.problem.instance_id}, "
                f"function: {self.problem.function_id})"
            )
        except Exception as e:
            raise ValueError(
                f"Failed to initialize COCO problem: {e}. "
                f"Suite options used: '{suite_options}'. "
                f"Available COCO suites: {cocoex.known_suite_names if hasattr(cocoex, 'known_suite_names') else 'Not available'}"
            ) from e

        if self.problem is None:
            raise ValueError(
                "COCO problem object is None after initialization attempt."
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts objectives using the initialized cocoex problem.

        Args:
            X: The input decision (numpy array). Expected to be 1D for a single sample
               or 2D (batch_size, n_features) for multiple samples.

        Returns:
            The predicted objectives (numpy array). Will be 2D (batch_size, 1) for scalar objectives.
        """
        # Ensure input is float64 as cocoex functions typically expect it
        processed_decision = X.astype(np.float64)

        if processed_decision.ndim == 2:
            # Apply the COCO problem function to each row in the batch
            results = np.array(
                [self.problem(d_single) for d_single in processed_decision]
            )
            # COCO functions typically return scalar objectives, so reshape to (batch_size, 1)
            return results.reshape(-1, 1).astype(np.float32)
        elif processed_decision.ndim == 1:
            # Apply the COCO problem function for a single input
            result = self.problem(processed_decision)
            # Ensure output is (1, 1) for a single scalar result
            return np.array([[result]]).astype(np.float32)
        else:
            raise ValueError(
                f"Unsupported input dimensions for COCO problem: {processed_decision.ndim}. Expected 1D or 2D (batch)."
            )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        This mapper does not require fitting as it's based on a predefined function.
        A warning is issued if this method is called.
        """
        print(
            "Warning: CocoexFunctionForwardDecisionMapper does not require fitting as it's based on a predefined function."
        )
