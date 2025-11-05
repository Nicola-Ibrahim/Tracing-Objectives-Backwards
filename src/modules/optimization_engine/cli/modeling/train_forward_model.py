from typing import Type

from ...application.dtos import (
    COCOEstimatorParams,
    CVAEEstimatorParams,
    CVAEMDNEstimatorParams,
    EstimatorParams,
    GaussianProcessEstimatorParams,
    MDNEstimatorParams,
    NeuralNetworkEstimatorParams,
    RBFEstimatorParams,
)
from .train_single_model import create_training_cli

FORWARD_ESTIMATOR_REGISTRY: dict[str, Type[EstimatorParams]] = {
    "coco": COCOEstimatorParams,
    "cvae": CVAEEstimatorParams,
    "cvae_mdn": CVAEMDNEstimatorParams,
    "gaussian_process": GaussianProcessEstimatorParams,
    "mdn": MDNEstimatorParams,
    "neural_network": NeuralNetworkEstimatorParams,
    "rbf": RBFEstimatorParams,
}


# Build a separate CLI for forward (decision -> objective) training workflows.
cli = create_training_cli(
    estimator_registry=FORWARD_ESTIMATOR_REGISTRY,
    mapping_direction="forward",
    help_prefix="a forward",
)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
