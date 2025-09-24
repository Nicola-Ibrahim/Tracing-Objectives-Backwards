import numpy as np

from ...application.factories.assurance import (
    create_conformal_calibrator,
    create_default_diversity_registry,
    create_default_scoring_strategy,
    create_forward_model,
    create_ood_calibrator,
)
from ...application.factories.estimator import EstimatorFactory
from ...application.modeling.generate_decision.generate_decision_command import (
    GenerateDecisionCommand,
)
from ...application.modeling.generate_decision.generate_decision_handler import (
    GenerateDecisionCommandHandler,
)
from ...domain.assurance.decision_validation import DecisionValidationService
from ...domain.assurance.feasibility import ObjectiveFeasibilityService
from ...domain.assurance.feasibility.value_objects import Tolerance
from ...infrastructure.loggers.cmd_logger import CMDLogger
from ...infrastructure.repositories.datasets.processed_dataset_repo import (
    FileSystemProcessedDatasetRepository,
)
from ...infrastructure.repositories.modeling.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)


def main():
    """
    Main function to generate a decision using hardcoded parameters.
    Modify the variables below to change the inputs.
    """

    # Create the command object using the hardcoded values
    command = GenerateDecisionCommand(
        estimator_type="mdn",
        target_objective=[411, 1500],
        distance_tolerance=0.02,
        num_suggestions=5,
        validation_enabled=False,
        feasibility_enabled=False,
    )

    # Instantiate domain services
    feasibility_service = ObjectiveFeasibilityService(
        scorer=create_default_scoring_strategy(),
        diversity_registry=create_default_diversity_registry(),
    )

    decision_validation_service: DecisionValidationService | None = None
    if command.validation_enabled:
        forward_model = EstimatorFactory().create_forward(
            {"type": "coco_biobj", "function_indices": 5}
        )
        tolerance = Tolerance(
            eps_l2=0.03,
            eps_per_obj=None,
        )
        forward_adapter = create_forward_model([forward_model])
        conformal_calibrator = (
            create_conformal_calibrator(confidence=0.90)
            if forward_adapter is not None
            else None
        )
        decision_validation_service = DecisionValidationService(
            tolerance=tolerance,
            ood_calibrator=create_ood_calibrator(
                percentile=97.5,
                cov_reg=1e-6,
            ),
            conformal_calibrator=conformal_calibrator,
            forward_model=forward_adapter,
        )

    # Initialize the handler with pre-built services
    handler = GenerateDecisionCommandHandler(
        model_repository=FileSystemModelArtifactRepository(),
        processed_data_repository=FileSystemProcessedDatasetRepository(),
        logger=CMDLogger(name="InterpolationCMDLogger"),
        feasibility_service=feasibility_service,
        decision_validation_service=decision_validation_service,
    )

    handler.execute(command)


if __name__ == "__main__":
    main()
