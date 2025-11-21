import click

from .train_inverse_model_common import (
    _build_inverse_cv_handler,
    _common_training_options,
    _create_estimator_params,
    _make_validation_metric_configs,
    _parse_overrides,
    TrainInverseModelCrossValidationCommand,
)


@click.command(help="Train inverse model with k-fold cross-validation")
@_common_training_options
@click.option(
    "--cv-splits",
    type=int,
    default=5,
    show_default=True,
    help="Number of cross-validation splits (must be > 1)",
)
def cli(
    estimator: str,
    params: tuple[str, ...],
    metrics: tuple[str, ...],
    random_state: int,
    learning_curve_steps: int,
    epochs: int,
    cv_splits: int,
) -> None:
    if cv_splits <= 1:
        raise click.BadParameter("must be greater than 1", param_hint="--cv-splits")

    handler = _build_inverse_cv_handler()
    overrides = _parse_overrides(params)
    estimator_params = _create_estimator_params(estimator, overrides)
    command = TrainInverseModelCrossValidationCommand(
        estimator_params=estimator_params,
        estimator_performance_metric_configs=_make_validation_metric_configs(metrics),
        random_state=random_state,
        cv_splits=cv_splits,
        learning_curve_steps=learning_curve_steps,
        epochs=epochs,
    )
    handler.execute(command)


def main() -> None:  # pragma: no cover - CLI entrypoint
    cli()


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
