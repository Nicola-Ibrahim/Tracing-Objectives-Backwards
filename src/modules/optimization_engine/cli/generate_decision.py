from ..application.interpolation.free_mode_generate_decision.free_mode_generate_decision_command import (
    FreeModeGenerateDecisionCommand,
)
from ..application.interpolation.free_mode_generate_decision.free_mode_generate_decision_handler import (
    FreeModeGenerateDecisionCommandHandler,
)
from ..domain.interpolation.enums.inverse_decision_mapper_type import (
    InverseDecisionMapperType,
)
from ..infrastructure.loggers.cmd_logger import CMDLogger
from ..infrastructure.repositories.generation.npz_pareto_data_repo import (
    NPZParetoDataRepository,
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
    interpolator_type_to_use = InverseDecisionMapperType.RBF_ND.value

    # Specify the target objective point (f1, f2, ...).
    # Modify this list with your desired values.
    target_objective_point = [415, 1200]

    # Initialize the handler with the repository
    handler = FreeModeGenerateDecisionCommandHandler(
        interpolation_model_repo=PickleInterpolationModelRepository(),
        pareto_data_repo=NPZParetoDataRepository(),
        logger=CMDLogger(name="InterpolationCMDLogger"),
    )

    # Create the command object using the hardcoded values
    command = FreeModeGenerateDecisionCommand(
        interpolator_type=interpolator_type_to_use,
        target_objective=target_objective_point,
        distance_tolerance=0.02,
        num_suggestions=5,
    )

    handler.execute(command)


if __name__ == "__main__":
    main()
