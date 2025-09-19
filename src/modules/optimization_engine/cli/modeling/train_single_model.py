import ast
from typing import Sequence

import click

from ...application.modeling.train_model.train_model_command import TrainModelCommand
from .common import (
    DEFAULT_VALIDATION_METRICS,
    ESTIMATOR_REGISTRY,
    build_processed_training_handler,
    create_estimator_params,
    make_validation_metric_configs,
)


def _literal(value: str):
    """Safely coerce CLI string values into Python literals when possible."""

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _parse_overrides(pairs: Sequence[str]) -> dict[str, object]:
    """Convert ``key=value`` pairs passed on the CLI into kwargs."""

    overrides: dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise click.BadParameter(
                f"Parameter override '{pair}' must be in the form key=value",
                param_hint="--param",
            )
        key, raw_value = pair.split("=", 1)
        overrides[key.strip()] = _literal(raw_value.strip())
    return overrides


def _common_options(func):
    options = [
        click.option(
            "--estimator",
            type=click.Choice(sorted(ESTIMATOR_REGISTRY.keys())),
            default="cvae",
            show_default=True,
            help="Which estimator configuration to use",
        ),
        click.option(
            "--param",
            "params",
            multiple=True,
            metavar="KEY=VALUE",
            help="Override estimator parameter (may be supplied multiple times)",
        ),
        click.option(
            "--metric",
            "metrics",
            multiple=True,
            default=DEFAULT_VALIDATION_METRICS,
            show_default=True,
            help="Validation metrics to report",
        ),
        click.option(
            "--random-state",
            type=int,
            default=42,
            show_default=True,
            help="Random seed for reproducibility",
        ),
        click.option(
            "--learning-curve-steps",
            type=int,
            default=50,
            show_default=True,
            help="Number of steps for deterministic learning curves",
        ),
        click.option(
            "--epochs",
            type=int,
            default=100,
            show_default=True,
            help="Epochs for probabilistic estimators",
        ),
    ]

    for option in reversed(options):
        func = option(func)
    return func


@click.group(help="Train a single model using different evaluation workflows")
def cli() -> None:
    return None


@cli.command(name="standard", help="Single train/test split training")
@_common_options
def command_standard(
    estimator: str,
    params: Sequence[str],
    metrics: Sequence[str],
    random_state: int,
    learning_curve_steps: int,
    epochs: int,
) -> None:
    handler = build_processed_training_handler()
    overrides = _parse_overrides(params)
    estimator_params = create_estimator_params(estimator, overrides)
    command = TrainModelCommand(
        estimator_params=estimator_params,
        estimator_performance_metric_configs=make_validation_metric_configs(metrics),
        random_state=random_state,
        cv_splits=1,
        learning_curve_steps=learning_curve_steps,
        epochs=epochs,
    )
    handler.execute(command)


@cli.command(name="cv", help="K-fold cross-validation without hyper-parameter tuning")
@_common_options
@click.option(
    "--cv-splits",
    type=int,
    default=5,
    show_default=True,
    help="Number of cross-validation splits (must be > 1)",
)
def command_cv(
    estimator: str,
    params: Sequence[str],
    metrics: Sequence[str],
    random_state: int,
    learning_curve_steps: int,
    epochs: int,
    cv_splits: int,
) -> None:
    if cv_splits <= 1:
        raise click.BadParameter(
            "must be greater than 1",
            param_hint="--cv-splits",
        )

    handler = build_processed_training_handler()
    overrides = _parse_overrides(params)
    estimator_params = create_estimator_params(estimator, overrides)
    command = TrainModelCommand(
        estimator_params=estimator_params,
        estimator_performance_metric_configs=make_validation_metric_configs(metrics),
        random_state=random_state,
        cv_splits=cv_splits,
        learning_curve_steps=learning_curve_steps,
        epochs=epochs,
    )
    handler.execute(command)


@cli.command(name="grid", help="Grid search with cross-validation")
@_common_options
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
    required=False,
    help="Estimator parameter name to tune",
)
@click.option(
    "--tune-param-value",
    "tune_param_values",
    multiple=True,
    metavar="VALUE",
    help="Candidate value for the tuned parameter (specify multiple times)",
)
def command_grid(
    estimator: str,
    params: Sequence[str],
    metrics: Sequence[str],
    random_state: int,
    learning_curve_steps: int,
    epochs: int,
    cv_splits: int,
    tune_param_name: str | None,
    tune_param_values: Sequence[str],
) -> None:
    if cv_splits <= 1:
        raise click.BadParameter(
            "must be greater than 1",
            param_hint="--cv-splits",
        )
    if not tune_param_name:
        raise click.BadParameter(
            "--tune-param-name is required", param_hint="--tune-param-name"
        )
    if not tune_param_values:
        raise click.BadParameter(
            "Provide at least one --tune-param-value",
            param_hint="--tune-param-value",
        )

    handler = build_processed_training_handler()
    overrides = _parse_overrides(params)
    estimator_params = create_estimator_params(estimator, overrides)
    command = TrainModelCommand(
        estimator_params=estimator_params,
        estimator_performance_metric_configs=make_validation_metric_configs(metrics),
        random_state=random_state,
        cv_splits=cv_splits,
        tune_param_name=tune_param_name,
        tune_param_range=[_literal(v) for v in tune_param_values],
        learning_curve_steps=learning_curve_steps,
        epochs=epochs,
    )
    handler.execute(command)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
