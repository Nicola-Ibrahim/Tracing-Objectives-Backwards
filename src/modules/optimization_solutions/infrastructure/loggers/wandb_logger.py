import pickle
import tempfile
from pathlib import Path
from typing import Any

import wandb

from ...domain.models.interfaces.base_logger import BaseLogger


class WandbLogger(BaseLogger):
    def __init__(
        self,
        project: str,
        entity: str | None = None,
        run_name: str | None = None,
        job_type: str = "training",
    ):
        """Initialize the W&B run."""
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            job_type=job_type,
        )

    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        """Log evaluation or training metrics."""
        self.run.log(metrics, step=step)

    def log_model(
        self,
        model: Any,
        name: str,
        model_type: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        notes: str | None = None,
        collection_name: str | None = None,
    ):
        """Log a model as a W&B artifact."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            pickle.dump(model, tmp_file)
            model_path = tmp_file.name

        artifact = wandb.Artifact(
            name=name,
            type="model",
            description=description or f"Model: {model_type or 'Unknown'}",
            metadata={
                "model_type": model_type,
                "parameters": parameters or {},
                "metrics": metrics or {},
                "notes": notes,
            },
        )
        artifact.add_file(model_path)
        self.run.log_artifact(artifact)

        if collection_name:
            self.run.link_artifact(
                artifact,
                f"wandb-registry-interpolation-models/{collection_name}",
            )

    def load_model(self, artifact_name: str) -> Any:
        """Download a model artifact and load the pickled model."""
        artifact = self.run.use_artifact(f"{artifact_name}:latest", type="model")
        model_dir = artifact.download()

        for file in Path(model_dir).glob("*.pkl"):
            with open(file, "rb") as f:
                return pickle.load(f)

        self.run.finish()

        raise FileNotFoundError("No .pkl file found in artifact.")
