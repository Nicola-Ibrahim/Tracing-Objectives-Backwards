import click

from .train_inverse_model_common import (
    _build_inverse_grid_handler,
    _common_training_options,
    _create_estimator_params,
    _literal,
    _make_validation_metric_configs,
    _parse_overrides,
    TrainInverseModelGridSearchCommand,
)


@click.command(help="Train inverse model with grid search + cross-validation")
@_common_training_options
@click.option(
    "--cv-splits",
    type=int,
    default=5,
    show_default=True,
    help="Number of cross-validation splits (must be > 1)",
)
@click.option(
    "--tune-param-name",
    type=str,
    required=True,
    help="Estimator parameter name to tune",
)
@click.option(
    "--tune-param-value",
    "tune_param_values",
    multiple=True,
    metavar="VALUE",
    help="Candidate value for the tuned parameter (specify multiple times)",
)
def cli(
    estimator: str,
    params: tuple[str, ...],
    metrics: tuple[str, ...],
    random_state: int,
    learning_curve_steps: int,
    epochs: int,
    cv_splits: int,
    tune_param_name: str,
    tune_param_values: tuple[str, ...],
) -> None:
    if cv_splits <= 1:
        raise click.BadParameter("must be greater than 1", param_hint="--cv-splits")
    if not tune_param_values:
        raise click.BadParameter(
            "Provide at least one --tune-param-value",
            param_hint="--tune-param-value",
        )

    handler = _build_inverse_grid_handler()
    overrides = _parse_overrides(params)
    estimator_params = _create_estimator_params(estimator, overrides)
    command = TrainInverseModelGridSearchCommand(
        estimator_params=estimator_params,
        estimator_performance_metric_configs=_make_validation_metric_configs(metrics),
        random_state=random_state,
        cv_splits=cv_splits,
        tune_param_name=tune_param_name,
        tune_param_range=[_literal(v) for v in tune_param_values],
        learning_curve_steps=learning_curve_steps,
        epochs=epochs,
    )
    handler.execute(command)


def main() -> None:  # pragma: no cover - CLI entrypoint
    cli()


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
