import numpy as np
import plotly.graph_objects as go

from ....domain.common.interfaces.base_logger import BaseLogger
from ....domain.datasets.entities.processed_dataset import ProcessedDataset
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ....domain.modeling.enums.estimator_type import EstimatorTypeEnum
from ....domain.modeling.interfaces.base_estimator import (
    BaseEstimator,
    ProbabilisticEstimator,
)
from ....domain.modeling.interfaces.base_repository import BaseModelArtifactRepository
from .command import GenerateDecisionCommand


class GenerateDecisionCommandHandler:
    """
    Generate Design Candidates (X) for a requested Objective (Y).
    
    Unified Handler for MDN and CVAE:
    - Removes external Feasibility/Validation services (redundant).
    - Adapts to model capabilities:
        - If MDN: Uses analytic `predict_topk` to find exact modes.
        - If CVAE: Uses `sample` to generate a cloud of options.
    """

    def __init__(
        self,
        model_repository: BaseModelArtifactRepository,
        processed_data_repository: BaseDatasetRepository,
        logger: BaseLogger,
    ):
        self._model_repository = model_repository
        self._processed_data_repo = processed_data_repository
        self._logger = logger

    def execute(self, command: GenerateDecisionCommand) -> dict[str, np.ndarray]:
        """
        Generate candidates for the target.
        Returns dictionary with 'candidates' (raw) and 'candidates_norm'.
        """
        # 1. Load Resources
        estimator = self._load_estimator(command.inverse_estimator_type)
        processed: ProcessedDataset = self._processed_data_repo.load(
            "dataset", variant="processed"
        )

        decisions_normalizer = processed.decisions_normalizer
        objectives_normalizer = processed.objectives_normalizer

        # 2. Normalize Target (Y)
        # command.target_objective is likely shape (y_dim,) -> (1, y_dim)
        target_y_raw = np.array(command.target_objective, dtype=float).reshape(1, -1)
        target_y_norm = objectives_normalizer.transform(target_y_raw)

        self._logger.log_info(f"Target Objective (Raw): {target_y_raw.tolist()}")

        # 3. Generate Candidates (Polymorphic Logic)
        candidates_norm = self._generate_candidates(
            estimator, 
            target_y_norm, 
            n_samples=command.n_samples 
        )


        # 4. Denormalize Results (X_norm -> X_raw)
        # Ensure 2D shape for inverse_transform: (N_samples, x_dim)
        if candidates_norm.ndim == 3:
            candidates_norm = candidates_norm.reshape(-1, candidates_norm.shape[-1])
            
        candidates_raw = decisions_normalizer.inverse_transform(candidates_norm)

        self._logger.log_info(f"candidates_raw shape: {candidates_raw.shape}")

        # 5. Verify with Forward Model (Optional but recommended for visualization)
        forward_estimator = self._load_estimator(
            command.forward_estimator_type, direction="forward"
        )
        
        # Predict objectives for the generated candidates
        # Forward model expects raw decisions (original space)
        predicted_objectives = forward_estimator.predict(candidates_raw)

        # 6. Visualize Results
        pareto_front = processed.pareto.front if processed.pareto else np.zeros((0, 2))
        self._visualize_results(
            pareto_front=pareto_front,
            target_objective=np.array(command.target_objective),
            predicted_objectives=predicted_objectives,
        )

        self._logger.log_info(f"Generated {len(candidates_raw)} candidates.")

        # 7. Return Results
        return {
            "decisions": candidates_raw,       
            "decisions_norm": candidates_norm, 
        }

    def _visualize_results(
        self, 
        pareto_front: np.ndarray, 
        target_objective: np.ndarray, 
        predicted_objectives: np.ndarray
    ):
        """
        Visualize the Target, Predicted Cloud, and Pareto Front.
        """
        fig = go.Figure()

        # 1. Pareto Front (Background Reference)
        fig.add_trace(
            go.Scatter(
                x=pareto_front[:, 0],
                y=pareto_front[:, 1],
                mode="markers",
                name="Pareto Front",
                marker=dict(color="lightgray", size=5, opacity=0.5),
            )
        )

        # 2. Predicted Objectives (Cloud)
        fig.add_trace(
            go.Scatter(
                x=predicted_objectives[:, 0],
                y=predicted_objectives[:, 1],
                mode="markers",
                name="Predicted Outcomes",
                marker=dict(color="blue", size=6, opacity=0.7),
            )
        )

        # 3. Target Objective (Star)
        fig.add_trace(
            go.Scatter(
                x=[target_objective[0]],
                y=[target_objective[1]],
                mode="markers",
                name="Target Objective",
                marker=dict(color="red", symbol="star", size=15, line=dict(width=2, color="black")),
            )
        )

        fig.update_layout(
            title="Decision Generation Analysis",
            xaxis_title="Objective 1",
            yaxis_title="Objective 2",
            template="plotly_white",
            width=1000,
            height=800,
        )
        
        fig.show()

    def _generate_candidates(
        self, 
        estimator: BaseEstimator, 
        target_y_norm: np.ndarray, 
        n_samples: int
    ) -> np.ndarray:
        """
        Strategy pattern to handle different probabilistic models.
        """

        # CASE A: It's a Probabilistic Model (MDN or CVAE)
        if isinstance(estimator, ProbabilisticEstimator):
            return estimator.sample(target_y_norm, n_samples=n_samples)
            


    def _load_estimator(
        self, 
        estimator_type: EstimatorTypeEnum, 
        direction: str = "inverse"
    ) -> BaseEstimator:
        artifact = self._model_repository.get_latest_version(
            estimator_type=estimator_type.value,
            mapping_direction=direction,
        )
        return artifact.estimator