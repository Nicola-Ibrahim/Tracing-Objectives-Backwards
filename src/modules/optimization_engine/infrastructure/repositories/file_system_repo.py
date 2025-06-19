import os

from ....shared.config import ROOT_PATH
from ...domain.interpolation.entities.interpolator_model import InterpolatorModel
from ...domain.interpolation.interfaces.base_repository import (
    BaseTrainedModelRepository,
)
from .serializers import FileSystemSerializer


class FileSystemTrainedModelRepository(BaseTrainedModelRepository):
    def __init__(self, model_serializer: FileSystemSerializer):
        self._base_save_path = ROOT_PATH / "models"
        self._model_serializer = model_serializer

    def save(self, model: InterpolatorModel) -> None:
        # Create a specific directory for this model
        model_dir = self._base_save_path / model.id
        os.makedirs(model_dir, exist_ok=True)

        # Save the fitted interpolator instance
        model_artifact_path = model_dir / "fitted_interpolator.joblib"
        self._model_serializer.save_model(
            model.fitted_interpolator, model_artifact_path
        )

        # Save the model metadata (excluding the fitted_interpolator itself to avoid recursion/duplication)
        metadata_path = model_dir / "metadata.json"
        model_metadata_dict = model.model_dump(
            exclude={"fitted_interpolator"}
        )  # For Pydantic v2
        # For Pydantic v1, use model.dict(exclude={"fitted_interpolator"})
        self._model_serializer.save_metadata(metadata_path, model_metadata_dict)

        print(f"Model '{model.name}' (ID: {model.id}) saved to {model_dir}")

    def get_by_id(self, model_id: str) -> InterpolatorModel:
        model_dir = self._base_save_path / model_id
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"Model with ID {model_id} not found at {model_dir}"
            )

        metadata_path = model_dir / "metadata.json"
        model_artifact_path = model_dir / "fitted_interpolator.joblib"

        # Load metadata
        metadata_dict = self._model_serializer.load_metadata(metadata_path)

        # Load fitted interpolator
        fitted_interpolator = self._model_serializer.load_model(model_artifact_path)

        # Reconstruct the InterpolatorModel entity.
        # Ensure 'parameters' is correctly re-instantiated into its Pydantic model type.
        # This might require some mapping logic, especially if 'parameters' can be of different types.
        # For simplicity, assuming the factory can help reconstruct.
        # A more robust solution might save the parameter type in metadata.

        # Example of re-instantiating parameters based on type if needed:
        # from ...application.dtos import NeuralNetworkInterpolatorParams, ...
        # if metadata_dict['interpolator_type'] == 'neural_network':
        #    params_instance = NeuralNetworkInterpolatorParams(**metadata_dict['parameters'])
        # else:
        #    params_instance = InterpolatorParams(**metadata_dict['parameters']) # Fallback or error

        # For this example, let's just re-pass the dict and let Pydantic handle it if possible
        # Or, assume 'parameters' field in InterpolatorModel constructor can take a dict

        # Recreate the Pydantic param model instance
        from ...application.train_model.dtos import (
            GeodesicInterpolatorParams,
            InterpolatorParams,
            KNearestNeighborInterpolatorParams,
            LinearInterpolatorParams,
            NeuralNetworkInterpolatorParams,
        )

        param_type_map = {
            "neural_network": NeuralNetworkInterpolatorParams,
            "geodesic": GeodesicInterpolatorParams,
            "k_nearest_neighbor": KNearestNeighborInterpolatorParams,
            "linear": LinearInterpolatorParams,
        }

        ActualParamModel = param_type_map.get(
            metadata_dict["interpolator_type"], InterpolatorParams
        )
        reconstructed_params = ActualParamModel(**metadata_dict["parameters"])

        return InterpolatorModel(
            id=metadata_dict["id"],
            name=metadata_dict["name"],
            interpolator_type=metadata_dict["interpolator_type"],
            parameters=reconstructed_params,  # Pass the reconstructed Pydantic model
            fitted_interpolator=fitted_interpolator,
            metrics=metadata_dict.get("metrics", {}),
            description=metadata_dict.get("description"),
            notes=metadata_dict.get("notes"),
            collection=metadata_dict.get("collection"),
        )
