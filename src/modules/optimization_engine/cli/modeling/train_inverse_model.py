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

INVERSE_ESTIMATOR_REGISTRY: dict[str, Type[EstimatorParams]] = {
    "cvae": CVAEEstimatorParams,
    "cvae_mdn": CVAEMDNEstimatorParams,
    "gaussian_process": GaussianProcessEstimatorParams,
    "mdn": MDNEstimatorParams,
    "neural_network": NeuralNetworkEstimatorParams,
    "rbf": RBFEstimatorParams,
    "coco": COCOEstimatorParams,
}

# Build a CLI tailored for inverse (objective -> decision) training workflows.
cli = create_training_cli(
    estimator_registry=INVERSE_ESTIMATOR_REGISTRY,
    mapping_direction="inverse",
    help_prefix="an inverse",
)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
