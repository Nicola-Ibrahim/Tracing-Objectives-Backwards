import click

from ....shared.infrastructure.archivers.npz import ParetoNPzArchiver
from ...application.analyzing_biobj_data.analyzing_biobj_data_command import (
    AnalyzeBiobjDataCommand,
)
from ...application.analyzing_biobj_data.analyzing_biobj_data_handler import (
    AnalyzeBiobjDataHandler,
)
from ...infrastructure.visualizers.plotly import PlotlyParetoVisualizer


def analyze_data():
    command = AnalyzeBiobjDataCommand()
    handler = AnalyzeBiobjDataHandler(ParetoNPzArchiver(), PlotlyParetoVisualizer())
    handler.execute(command)


if __name__ == "__main__":
    analyze_data()
