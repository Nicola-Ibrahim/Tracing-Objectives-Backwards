from ...generating.domain.entities.pareto_data import ParetoDataModel
from ...shared.adapters.archivers.npz import ParetoNPzArchiver
from ..adapters.visualization import plot_pareto_visualizations


def analyze_pareto_data():
    # Load saved data
    archiver = ParetoNPzArchiver()
    raw_data: ParetoDataModel = archiver.load(filename="pareto_data.npz")

    # Access individual components
    X_data = raw_data.pareto_set
    Y_data = raw_data.pareto_front

    plot_pareto_visualizations(
        X_data,
        Y_data,
    )
