from ...application.factories.assurance import (
    create_default_diversity_registry,
    create_default_scoring_strategy,
)
from ...application.modeling.generate_decision.generate_decision_command import (
    GenerateDecisionCommand,
)
from ...application.modeling.generate_decision.generate_decision_handler import (
    GenerateDecisionCommandHandler,
)
from ...domain.assurance.decision_validation import DecisionValidationService
from ...domain.assurance.feasibility import ObjectiveFeasibilityService
from ...domain.assurance.feasibility.value_objects import Tolerance
from ...infrastructure.datasets.repositories.processed_dataset_repo import (
    FileSystemProcessedDatasetRepository,
)
from ...infrastructure.loggers.cmd_logger import CMDLogger
from ...infrastructure.modeling.repositories.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)
from ...infrastructure.repositories.assurance import (
    FileSystemDecisionValidationCalibrationRepository,
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
        validation_enabled=True,
        feasibility_enabled=False,
    )

    # Instantiate domain services
    feasibility_service = ObjectiveFeasibilityService(
        scorer=create_default_scoring_strategy(),
        diversity_registry=create_default_diversity_registry(),
    )

    tolerance = Tolerance(
        eps_l2=0.03,
        eps_per_obj=None,
    )
    decision_validation_service = DecisionValidationService(
        tolerance=tolerance,
        calibration_repository=FileSystemDecisionValidationCalibrationRepository(),
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
