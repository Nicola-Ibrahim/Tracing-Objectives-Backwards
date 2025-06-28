import datetime
import json
import os
import pickle
from pathlib import Path
from typing import Any

from .....shared.config import ROOT_PATH
from ....application.interpolation.train_model.dtos import (
    GeodesicInterpolatorParams,
    LinearInverseDecisionMapperParams,
    NearestNeighborInverseDecisoinMapperParams,
    NeuralNetworkInverserDecisionMapperParams,
)
from ....domain.interpolation.entities.interpolator_model import InterpolatorModel
from ....domain.interpolation.interfaces.base_inverse_decision_mappers import (
    BaseInverseDecisionMapper,
)
from ....domain.interpolation.interfaces.base_repository import (
    BaseInterpolationModelRepository,
)


class DateTimeEncoder(json.JSONEncoder):
    """
    Custom JSONEncoder to serialize datetime objects to ISO 8601 format.
    """

    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        return super().default(obj)


class InverseDecisionMapperFileHandler:
    """
    Handles the serialization and deserialization of model artifacts (using pickle)
    and their associated metadata (using JSON) to and from the file system.
    """

    def save(self, inverse_decsion_mapper: BaseInverseDecisionMapper, file_path: Path):
        """Saves a model instance to a specified file path using pickle."""
        try:
            with open(file_path, "wb") as f:
                pickle.dump(inverse_decsion_mapper, f)
        except Exception as e:
            raise IOError(f"Failed to pickle model instance to {file_path}: {e}") from e

    def load(self, file_path: Path) -> BaseInverseDecisionMapper:
        """Loads a model instance from a specified file path using pickle."""
        if not file_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {file_path}")
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise IOError(
                f"Failed to unpickle model instance from {file_path}: {e}"
            ) from e

    def save_metadata(self, file_path: Path, metadata_content: dict[str, Any]):
        """Saves model metadata as a JSON file to a specified path."""
        try:
            with open(file_path, "w") as f:
                json.dump(metadata_content, f, indent=4, cls=DateTimeEncoder)
        except Exception as e:
            raise IOError(f"Failed to save metadata to {file_path}: {e}") from e

    def load_metadata(self, file_path: Path) -> dict[str, Any]:
        """Loads model metadata from a JSON file at a specified path."""
        if not file_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {file_path}")
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            raise IOError(f"Failed to load metadata from {file_path}: {e}") from e


class PickleInterpolationModelRepository(BaseInterpolationModelRepository):
    """
    Manages the persistence of InterpolatorModel entities, including version tracking,
    using the file system for storage. Each model version (identified by its unique ID)
    is stored in its own dedicated directory.
    """

    def __init__(self):
        self._base_model_storage_path = ROOT_PATH / "models"
        self._inverse_decision_mapper_handler = InverseDecisionMapperFileHandler()
        self._base_model_storage_path.mkdir(
            parents=True, exist_ok=True
        )  # Ensure the base models directory exists

        # Registry to map string types to their corresponding Pydantic parameter models for deserialization
        self._parameter_model_registry = {
            "neural_network": NeuralNetworkInverserDecisionMapperParams,
            "geodesic": GeodesicInterpolatorParams,
            "k_nearest_neighbor": NearestNeighborInverseDecisoinMapperParams,
            "linear": LinearInverseDecisionMapperParams,
        }

    def save(self, interpolator_model: InterpolatorModel) -> None:
        """
        Saves a new InterpolatorModel entity, representing a specific training version.
        A dedicated directory is created for this model version using its unique ID.

        Args:
            interpolator_model: The InterpolatorModel entity to save. Its 'id' field
                                  is used to determine the storage location.
        """
        if not interpolator_model.id:
            raise ValueError(
                "InterpolatorModel entity must have a unique 'id' for saving."
            )

        # --- NEW FILE STRUCTURE LOGIC ---
        # Create a subdirectory for the interpolator type
        model_type_directory = (
            self._base_model_storage_path / interpolator_model.parameters.get("type")
        )

        model_type_directory.mkdir(exist_ok=True)

        # Then create a directory for the unique model ID within the type directory
        model_version_directory = model_type_directory / interpolator_model.id
        os.makedirs(model_version_directory, exist_ok=True)

        # Define file paths within the model's dedicated directory
        mapper_artifact_path = model_version_directory / "inverse_decision_mapper.pkl"
        metadata_file_path = model_version_directory / "metadata.json"

        # Save the inverse decision mapper instance and metadata
        self._inverse_decision_mapper_handler.save(
            interpolator_model.inverse_decision_mapper, mapper_artifact_path
        )
        model_metadata = interpolator_model.to_save_format()
        self._inverse_decision_mapper_handler.save_metadata(
            metadata_file_path, model_metadata
        )

    def load(self, model_version_id: str) -> InterpolatorModel:
        """
        Retrieves a specific InterpolatorModel entity by its unique version ID.

        Args:
            model_version_id: The unique identifier of the specific model version to load.

        Returns:
            The loaded InterpolatorModel entity.

        Raises:
            FileNotFoundError: If the model version with the specified ID is not found.
            IOError: For other loading errors (e.g., corrupted files).
        """
        model_version_directory = self._base_model_storage_path / model_version_id
        if not model_version_directory.exists():
            raise FileNotFoundError(
                f"Model version with ID {model_version_id} not found at {model_version_directory}"
            )

        metadata_file_path = model_version_directory / "metadata.json"
        mapper_artifact_path = model_version_directory / "inverse_decision_mapper.pkl"

        # Load metadata
        model_metadata = self._inverse_decision_mapper_handler.load_metadata(
            metadata_file_path
        )

        # Load the fitted inverse decision mapper
        loaded_decision_mapper = self._inverse_decision_mapper_handler.load(
            mapper_artifact_path
        )

        return InterpolatorModel.from_saved_format(
            saved_data=model_metadata, loaded_mapper=loaded_decision_mapper
        )

    def get_all_versions_by_conceptual_name(
        self, model_conceptual_name: str
    ) -> list[InterpolatorModel]:
        """
        Retrieves all trained versions of a model that share a given conceptual name.

        Args:
            model_conceptual_name: The conceptual name of the model type (e.g., 'f1_vs_f2_PchipMapper').

        Returns:
            A list of InterpolatorModel entities, sorted by 'trained_at' timestamp in descending order (latest first).
        """
        found_model_versions: list[InterpolatorModel] = []
        for model_version_directory in self._base_model_storage_path.iterdir():
            if model_version_directory.is_dir():
                try:
                    metadata_file_path = model_version_directory / "metadata.json"
                    model_metadata = (
                        self._inverse_decision_mapper_handler.load_metadata(
                            metadata_file_path
                        )
                    )

                    if model_metadata.get("name") == model_conceptual_name:
                        # Load the full model entity by its ID (directory name)
                        # The load method now handles all the deserialization logic
                        model_version = self.load(model_version_directory.name)
                        found_model_versions.append(model_version)
                except (
                    FileNotFoundError,
                    json.JSONDecodeError,
                    KeyError,
                    ValueError,
                ) as e:
                    print(
                        f"Warning: Could not process directory '{model_version_directory.name}': {e}. Skipping."
                    )
                    continue

        found_model_versions.sort(key=lambda model: model.trained_at, reverse=True)
        return found_model_versions

    def get_latest_version(self, model_conceptual_name: str) -> InterpolatorModel:
        """
        Retrieves the latest trained version of a model based on its conceptual name.
        The 'latest' version is determined by the most recent 'trained_at' timestamp.

        Args:
            model_conceptual_name: The conceptual name of the model type.

        Returns:
            The InterpolatorModel entity representing the latest version.

        Raises:
            FileNotFoundError: If no model versions are found for the given conceptual name.
            Exception: For other errors during version lookup.
        """
        found_model_versions = self.get_all_versions_by_conceptual_name(
            model_conceptual_name
        )

        if not found_model_versions:
            raise FileNotFoundError(
                f"No model versions found for conceptual name: '{model_conceptual_name}'"
            )

        # The list is already sorted by 'trained_at' in descending order, so the first element is the latest.
        return found_model_versions[0]
