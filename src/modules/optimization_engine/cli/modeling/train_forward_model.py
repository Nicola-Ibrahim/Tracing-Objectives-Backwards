from ...application.modeling.train_forward_model import (
    TrainForwardModelCommand,
)
from ..common import (
    FORWARD_ESTIMATOR_REGISTRY,
    build_forward_training_handler,
)
from .train_single_model import create_training_cli


# Build a separate CLI for forward (decision -> objective) training workflows.
cli = create_training_cli(
    estimator_registry=FORWARD_ESTIMATOR_REGISTRY,
    command_cls=TrainForwardModelCommand,
    handler_builder=build_forward_training_handler,
    help_prefix="a forward",
)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
