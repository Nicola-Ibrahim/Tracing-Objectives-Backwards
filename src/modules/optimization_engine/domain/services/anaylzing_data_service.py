from pathlib import Path

from ...infrastructure.acl.pareto_generation_acl import (
    AnalysisResultDTO,
    GenerationDataACL,
)


class BiobjAnalysisDataService:
    """
    Service within the 'analyzing' bounded context responsible for
    orchestrating the retrieval and preparation of biobjective Pareto data
    for analysis. It uses the ACL to communicate with the 'optimization' context.
    """

    def __init__(self, acl: GenerationDataACL):
        """
        Initializes the service with a dependency on the GenerationDataACL.

        Args:
            acl: The Anti-Corruption Layer instance.
        """
        self._acl = acl

    def get_biobj_analysis_data(self, data_identifier: Path | str) -> AnalysisResultDTO:
        """
        Retrieves biobjective Pareto data via the ACL and returns it in
        the 'analyzing' context's preferred DTO format.

        This method acts as a dedicated entry point for retrieving analysis-ready data.

        Args:
            data_identifier: The identifier (e.g., file path) of the Pareto data.

        Returns:
            AnalysisResultDTO: The transformed Pareto data suitable for analysis.

        Raises:
            FileNotFoundError: If the data specified by `data_identifier` is not found
                               (propagated from the ACL/Archiver).
            Exception: Any other exception during data retrieval or transformation.
        """
        print(
            f"BiobjAnalysisDataService: Requesting data for '{data_identifier}' from ACL."
        )
        # The service calls the ACL
        analysis_data: AnalysisResultDTO = self._acl.get_pareto_analysis_data(
            data_identifier
        )
        print(
            "BiobjAnalysisDataService: Data successfully retrieved and transformed by ACL."
        )
        return analysis_data
