from pathlib import Path
from typing import Any

from .base import BaseFileHandler

from safetensors.torch import load_file, save_file


class SafeTensorsFileHandler(BaseFileHandler):
    """Handles serialization of torch tensor state dicts using safetensors."""

    def save(self, obj: Any, file_path: Path):
        save_file(obj, str(file_path))

    def load(self, file_path: Path) -> Any:
        if not file_path.exists():
            raise FileNotFoundError(f"Artifact not found at {file_path}")
        return load_file(str(file_path))
