import json
from typing import Any

from ....shared.config import ROOT_PATH


class InterpolatorsMetricsLoader:
    """
    Loads specific metrics from trained model directories.
    This is a data access layer component within the Infrastructure Layer.
    """

    def __init__(self):
        """
        Args:
            interpolators_dir (Path): The root directory containing all trained models (e.g., 'models/').
        """
        self.interpolators_dir = None

    def fetch(self, dir_name: str) -> list[dict[str, Any]]:
        """
        Traverses the models directory and extracts model type and MSE from each metadata.json file.

        Returns:
            list[dict[str, Any]]: A list of dictionaries, e.g., [{'model_type': 'gaussian_process_nd', 'mse': 0.0176}].
        """

        interpolators_dir = ROOT_PATH / dir_name
        if not interpolators_dir.is_dir():
            raise FileNotFoundError(f"Models directory not found: {interpolators_dir}")
        self.interpolators_dir = interpolators_dir

        grouped_metrics: dict[str, list[float]] = {}

        # In a real setup, you would import ModelMetadata here.
        # For demonstration without showing the model, we use direct dictionary access.
        # from src.application.models import ModelMetadata # (Uncomment in real code)

        for metadata_file in self.interpolators_dir.glob("**/metadata.json"):
            try:
                # Read the raw JSON content
                metadata_content = metadata_file.read_text()
                raw_data = json.loads(metadata_content)

                # Extract the required fields directly from the raw dictionary
                model_type = raw_data.get("parameters", {}).get("type")
                mse_metric = raw_data.get("metrics", {}).get(
                    "MeanSquaredErrorValidationMetric"
                )

                if model_type and mse_metric is not None:
                    if model_type not in grouped_metrics:
                        grouped_metrics[model_type] = []
                    grouped_metrics[model_type].append(mse_metric)
                else:
                    print(
                        f"MetricsLoader: Missing required fields in {metadata_file}. Skipping."
                    )

            except json.JSONDecodeError as e:
                print(
                    f"MetricsLoader: Error decoding JSON from {metadata_file}. Skipping. Error: {e}"
                )
            except Exception as e:
                print(
                    f"MetricsLoader: Unexpected error parsing {metadata_file}. Skipping. Error: {e}"
                )

        return grouped_metrics
