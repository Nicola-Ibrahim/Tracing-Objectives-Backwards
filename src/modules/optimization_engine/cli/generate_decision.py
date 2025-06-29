from ..application.interpolation.free_mode_generate_decision.free_mode_generate_decision_command import (
    FreeModeGenerateDecisionCommand,
)
from ..application.interpolation.free_mode_generate_decision.free_mode_generate_decision_handler import (
    FreeModeGenerateDecisionCommandHandler,
)
from ..domain.interpolation.enums.inverse_decision_mapper_type import (
    InverseDecisionMapperType,
)
from ..infrastructure.repositories.interpolation.pickle_interpolator_repo import (
    PickleInterpolationModelRepository,
)


def main():
    """
    Main function to generate a decision using hardcoded parameters.
    Modify the variables below to change the inputs.
    """
    # Specify the type of interpolator to use.
    interpolator_type_to_use = InverseDecisionMapperType.GAUSSIAN_PROCESS_ND.value

    # Specify the target objective point (f1, f2, ...).
    # Modify this list with your desired values.
    target_objective_point = [410, 1240]

    # Initialize the repository
    interpolation_model_repo = PickleInterpolationModelRepository()

    # Initialize the handler with the repository
    handler = FreeModeGenerateDecisionCommandHandler(
        interpolation_model_repo=interpolation_model_repo
    )

    # Create the command object using the hardcoded values
    command = FreeModeGenerateDecisionCommand(
        interpolator_type=interpolator_type_to_use,
        target_objective=target_objective_point,
    )

    # Execute the command
    predicted_decision = handler.execute(command)

    print("\n--- Prediction Result ---")
    print(f"Interpolator Type: {interpolator_type_to_use}")
    print(f"Target Objective: {target_objective_point}")
    print(f"Predicted Decision (X-values): {predicted_decision.tolist()}")


if __name__ == "__main__":
    main()
