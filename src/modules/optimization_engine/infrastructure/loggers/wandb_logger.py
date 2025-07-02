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

    def log_error(self, message: str) -> None:
        """Log an error message to W&B."""
        self.run.log({"error_message": message})
        print(f"Wandb Error: {message}")  # Also print to console for immediate feedback

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
        step: Optional[
            int
        ] = None,  # Now matches BaseLogger - W&B uses this in log() for metrics, not directly for artifacts
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
            # or `entity/project/collection_name`. Assuming `wandb-registry-interpolation-models`
            # is meant to be the entity or part of the project name for linking.
            # A common pattern is `project_name/model_registry_name`.
            # Let's assume collection_name is just the specific model collection within your project.
            # The format for link_artifact is typically "entity/project/artifact_name" or "project/artifact_name"
            # It seems you want to link to a W&B Model Registry.
            # The target for link_artifact should be a fully qualified path to the registry
            # e.g., "model_registry/MyAwesomeModels"
            # If "wandb-registry-interpolation-models" is a specific project or entity, this might work.
            # A more common registry linking would be:
            # self.run.link_artifact(artifact, f"{self.run.project}/model_registry/{collection_name}")
            # Or simpler if `collection_name` is intended to be the alias for this artifact version:
            self.run.link_artifact(
                artifact,
                f"wandb-artifact://{self.run.project}/{name}",  # Link the specific artifact to itself but with a new alias/collection
                aliases=[collection_name],
            )
            # If `collection_name` is meant to be a model name in the registry, you'd do:
            # self.run.link_artifact(artifact, collection_name) # This would push the model to a registry with `collection_name` as the name
            # For the original structure:
            print(f"Wandb: Linking artifact to collection: {collection_name}")

        # Clean up the temporary file after logging
        Path(model_path).unlink(missing_ok=True)

    def load_model(self, artifact_name: str) -> Any:
        """Download a model artifact and load the pickled model."""
        # Using self.run.use_artifact means the run needs to be active.
        # Consider if you want to load models without an active run, which would require
        # wandb.Api().artifact(...)
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
            # Do not call run.finish() here, as this method might be called multiple times
            # within an active run. The `run.finish()` should be handled externally
            # when the entire logging process for a run is complete.
            pass
