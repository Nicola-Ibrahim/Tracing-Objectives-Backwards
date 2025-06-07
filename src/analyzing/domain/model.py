import pickle
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Self

import wandb
from pydantic import BaseModel, Field


class InterpolatorModel(BaseModel):
    name: str = Field(..., description="Model name used for tracking")
    model: Any = Field(..., description="Fitted interpolator object")
    notes: str | None = Field(None, description="Free-form notes about the model")
    model_type: str | None = Field(None, description="Type or class name of the model")
    created_at: str | None = Field(
        default_factory=lambda: datetime.now(datetime.timezone.utc)
    )

    def save_to_wandb(
        self,
        project: str = "interpolation-models",
        entity: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
    ):
        """Log model and metadata to Weights & Biases as an artifact."""
        wandb.init(
            project=project, entity=entity, name=self.name, job_type="save-model"
        )

        # Save model temporarily
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            pickle.dump(self.model, tmp_file)
            tmp_file_path = tmp_file.name

        # Create artifact
        artifact = wandb.Artifact(
            name=self.name,
            type="model",
            description=description or f"Model: {self.model_type or 'Unknown'}",
            metadata={
                "name": self.name,
                "model_type": self.model_type,
                "created_at": self.created_at,
                "notes": self.notes,
                "parameters": parameters or {},
                "metrics": metrics or {},
            },
        )

        artifact.add_file(tmp_file_path)
        wandb.log_artifact(artifact)
        wandb.finish()

    @classmethod
    def load_from_wandb(
        cls,
        artifact_name: str,
        project: str = "interpolation-models",
        entity: str | None = None,
    ) -> Self:
        """Load a model and its metadata from W&B."""
        wandb.init(project=project, entity=entity, job_type="load-model")

        artifact = wandb.use_artifact(f"{artifact_name}:latest", type="model")
        model_path = artifact.download()

        # Load model file
        for file in Path(model_path).glob("*.pkl"):
            with open(file, "rb") as f:
                model = pickle.load(f)
            break
        else:
            raise FileNotFoundError("No .pkl model file found in the W&B artifact.")

        # Extract metadata and build the model
        metadata = artifact.metadata
        metadata.update({"model": model, "name": artifact.name})

        wandb.finish()
        return cls(**metadata)
