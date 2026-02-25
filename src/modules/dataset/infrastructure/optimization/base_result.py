from abc import ABC, abstractmethod


class BaseResultProcessor(ABC):
    def __init__(self) -> None:
        """
        Base class for processing optimization results.
        This class should be extended to implement specific result processing logic.
        """
        pass

    @abstractmethod
    def process(self, result) -> None:
        """
        Process the optimization results to ensure they are ready for extraction.
        This method should be implemented by subclasses to handle specific processing logic.

        Args:
            result: The optimization result to process.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_history(self) -> list:
        """
        Extract optimization history with domain-specific key naming.

        Returns:
            A list of dictionaries containing the optimization history.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_full_solutions(self) -> dict:
        """
        Get complete solution set with domain-specific naming.

        Returns:
            A dictionary containing the full solutions.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_solution_set(self) -> dict:
        """
        Get the solution set with domain-specific naming.

        Returns:
            A dictionary containing the solution set.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_pareto_front(self) -> dict:
        """
        Extract the Pareto front from the optimization results.

        Returns:
            A dictionary containing the Pareto front solutions.
        """
        raise NotImplementedError("Subclasses must implement this method.")
