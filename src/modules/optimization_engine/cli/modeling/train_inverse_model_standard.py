import click

from .train_inverse_model_common import (
    _build_inverse_training_handler,
    _common_training_options,
    _create_estimator_params,
    _make_validation_metric_configs,
    _parse_overrides,
    TrainInverseModelCommand,
)


@click.command(help="Train inverse model using a single train/test split")
@_common_training_options
def cli(
    estimator: str,
    params: tuple[str, ...],
    metrics: tuple[str, ...],
    random_state: int,
    learning_curve_steps: int,
    epochs: int,
) -> None:
    handler = _build_inverse_training_handler()
    overrides = _parse_overrides(params)
    estimator_params = _create_estimator_params(estimator, overrides)
    command = TrainInverseModelCommand(
        estimator_params=estimator_params,
        estimator_performance_metric_configs=_make_validation_metric_configs(metrics),
        random_state=random_state,
        learning_curve_steps=learning_curve_steps,
        epochs=epochs,
    )
    handler.execute(command)


def main() -> None:  # pragma: no cover - CLI entrypoint
    cli()


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
