import click

from ...shared.adapters.archivers.npz import ParetoNPzArchiver
from ..infrastructure.plotters.plotly import PlotlyParetoPlotter
from ..application.analyzing_biobj_data.analyzing_biobj_data_command import (
    AnalyzeBiobjDataCommand,
)
from ..application.analyzing_biobj_data.analyzing_biobj_data_handler import (
    AnalyzeBiobjDataHandler,
)


@click.command()
@click.option("--results-path", required=True, help="Path to generated results.")
@click.option(
    "--output-path", required=False, default="plots/", help="Where to save plots."
)
def analyze_data(results_path: str, output_path: str):
    command = AnalyzeBiobjDataCommand(
        results_path=results_path, output_path=output_path
    )
    handler = AnalyzeBiobjDataHandler(ParetoNPzArchiver(), PlotlyParetoPlotter())
    handler.execute(command)


if __name__ == "__main__":
    analyze_data()
