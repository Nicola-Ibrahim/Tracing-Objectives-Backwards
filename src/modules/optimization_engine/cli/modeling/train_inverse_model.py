from ...application.modeling.train_inverse_model import TrainInverseModelCommand
from ..common import (
    INVERSE_ESTIMATOR_REGISTRY,
    build_inverse_training_handler,
)
from .train_single_model import create_training_cli

# Build a CLI tailored for inverse (objective -> decision) training workflows.
cli = create_training_cli(
    estimator_registry=INVERSE_ESTIMATOR_REGISTRY,
    command_cls=TrainInverseModelCommand,
    handler_builder=build_inverse_training_handler,
    help_prefix="an inverse",
)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
