import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import wandb

from ...domain.interpolation.interfaces.base_logger import BaseLogger


class WandbLogger(BaseLogger):
    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        job_type: str = "training",
    ):
        """Initialize the W&B run."""
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            job_type=job_type,
            reinit=True,  # Important for notebook/repeated script runs
        )

    def log_info(self, message: str) -> None:
        """Log an informational message to W&B (as a print or custom log)."""
        # W&B captures stdout/stderr by default, so direct print might be sufficient.
        # For more structured logs, you could log it as a W&B event:
        self.run.log({"info_message": message})
        print(f"Wandb Info: {message}")  # Also print to console for immediate feedback

    def log_warning(self, message: str) -> None:
        """Log a warning message to W&B."""
        # W&B does not have a direct 'warning' log type like standard loggers.
        # We can log it as a custom event or as part of the system logs if W&B captures stderr.
        self.run.log({"warning_message": message})
        print(
            f"Wandb Warning: {message}"
        )  # Also print to console for immediate feedback

    def log_error(self, message: str) -> None:
        """Log an error message to W&B."""
        self.run.log({"error_message": message})
        print(f"Wandb Error: {message}")  # Also print to console for immediate feedback

    def log_debug(self, message: str) -> None:
        """Log a debug message to W&B."""
        # Similar to warning, W&B doesn't have a dedicated 'debug' log.
        # Log it as an event, but typically debug messages are very verbose
        # and might not be logged to W&B unless specifically needed for analysis.
        # For this example, we'll log it as a custom event.
        self.run.log({"debug_message": message})
        print(f"Wandb Debug: {message}")  # Also print to console for immediate feedback

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log evaluation or training metrics."""
        self.run.log(metrics, step=step)

    def log_model(
        self,
        model: Any,
        name: str,
        model_type: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        notes: Optional[str] = None,
        collection_name: Optional[str] = None,
        step: Optional[int] = None,
    ):
        """Log a model as a W&B artifact."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            pickle.dump(model, tmp_file)
            model_path = tmp_file.name

        # Prepare metadata for the artifact
        metadata = {
            "model_type": model_type,
            "parameters": parameters or {},
            "metrics": metrics or {},
            "notes": notes,
        }
        if step is not None:  # Include step in metadata if provided
            metadata["step"] = step

        artifact = wandb.Artifact(
            name=name,
            type="model",
            description=description or f"Model: {model_type or 'Unknown'}",
            metadata=metadata,
        )
        artifact.add_file(model_path)
        self.run.log_artifact(artifact)

        if collection_name:
            # For linking to a registry, the name should typically be `project/collection_name`
            # or `entity/project/collection_name`.
            # A common pattern for linking an artifact to a Model Registry entry is:
            # self.run.link_artifact(artifact, f"{self.run.project}/{collection_name}")
            # Or if `collection_name` is intended as an alias for the artifact:
            self.run.link_artifact(
                artifact,
                f"wandb-artifact://{self.run.project}/{name}",
                aliases=[collection_name],
            )
            print(f"Wandb: Linking artifact to collection/alias: {collection_name}")

        # Clean up the temporary file after logging
        Path(model_path).unlink(missing_ok=True)

    def load_model(self, artifact_name: str) -> Any:
        """Download a model artifact and load the pickled model."""
        try:
            artifact = self.run.use_artifact(f"{artifact_name}:latest", type="model")
            model_dir = artifact.download()

            for file in Path(model_dir).glob("*.pkl"):
                with open(file, "rb") as f:
                    model = pickle.load(f)
                    print(f"Wandb: Successfully loaded model from {file}")
                    return model
            raise FileNotFoundError("No .pkl file found in artifact.")
        except Exception as e:
            self.log_error(f"Failed to load model artifact {artifact_name}: {e}")
            raise
        finally:
            pass
