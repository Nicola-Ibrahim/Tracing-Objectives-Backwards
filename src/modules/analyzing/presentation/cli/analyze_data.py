from ....optimization_solutions.domain.services.pareto_generation_service import (
    ParetoGenerationService,
)
from ....optimization_solutions.infrastructure.algorithms import AlgorithmFactory
from ....optimization_solutions.infrastructure.archivers.npz import ParetoNPzArchiver
from ....optimization_solutions.infrastructure.optimizers import OptimizerFactory
from ....optimization_solutions.infrastructure.problems import ProblemFactory
from ...application.analyzing_biobj_data.analyzing_biobj_data_command import (
    AnalyzeBiobjDataCommand,
)
from ...application.analyzing_biobj_data.analyzing_biobj_data_handler import (
    AnalyzeBiobjDataCommandHandler,
)
from ...domain.service.anaylzing_data_service import BiobjAnalysisDataService
from ...infrastructure.acl.pareto_generation_acl import GenerationDataACL
from ...infrastructure.visualizers.plotly import PlotlyParetoVisualizer


def analyze_data():
    # --- Instantiate Optimization Context dependencies (as done in the main block before) ---
    archiver = ParetoNPzArchiver()
    problem_factory = ProblemFactory()
    algorithm_factory = AlgorithmFactory()
    optimizer_factory = OptimizerFactory()

    # Correctly instantiate ParetoGenerationService with its dependencies
    pareto_generation_service = ParetoGenerationService(
        problem_factory=problem_factory,
        algorithm_factory=algorithm_factory,
        optimizer_factory=optimizer_factory,
        archiver=archiver,
    )

    # --- Use the correctly instantiated service in the ACL ---
    command = AnalyzeBiobjDataCommand()
    handler = AnalyzeBiobjDataCommandHandler(
        analysis_data_service=BiobjAnalysisDataService(
            acl=GenerationDataACL(pareto_generation_service=pareto_generation_service)
        ),
        visualizer=PlotlyParetoVisualizer(),
    )

    handler.execute(command)


if __name__ == "__main__":
    # For a full demonstration, you'd first need to generate some data
    # using pareto_generation_service.generate_pareto_data(...)
    # to get a valid 'filename' or 'data_identifier' to pass to AnalyzeBiobjDataCommand.
    # The example in the previous turn's __main__ block shows this full flow.

    # This analyze_data() function is now conceptually correct in its dependency wiring,
    # but still needs a valid 'filename' argument for AnalyzeBiobjDataCommand
    # and a proper PlotlyParetoVisualizer implementation.
    analyze_data()
