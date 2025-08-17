from ..application.model_management.generate_decision.generate_decision_command import (
    GenerateDecisionCommand,
)
from ..application.model_management.generate_decision.generate_decision_handler import (
    GenerateDecisionCommandHandler,
)
from ..domain.model_management.enums.inverse_decision_mapper_type import (
    InverseDecisionMapperType,
)
from ..infrastructure.loggers.cmd_logger import CMDLogger
from ..infrastructure.repositories.generation.npz_pareto_data_repo import (
    NPZParetoDataRepository,
)
from ..infrastructure.repositories.model_management.pickle_model_artifact_repo import (
    PickleInterpolationModelRepository,
)


def main():
    """
    Main function to generate a decision using hardcoded parameters.
    Modify the variables below to change the inputs.
    """

    # Specify the target objective point (f1, f2, ...).
    target_objective_point = [411, 1242]

    # Initialize the handler with the repository
    handler = GenerateDecisionCommandHandler(
        interpolation_model_repo=PickleInterpolationModelRepository(),
        pareto_data_repo=NPZParetoDataRepository(),
        logger=CMDLogger(name="InterpolationCMDLogger"),
    )

    # Create the command object using the hardcoded values
    command = GenerateDecisionCommand(
        interpolator_type=InverseDecisionMapperType.KRIGING_ND.value,
        target_objective=target_objective_point,
        distance_tolerance=0.02,
        num_suggestions=5,
    )

    handler.execute(command)


if __name__ == "__main__":
    main()
