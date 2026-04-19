from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .base import BaseFileHandler


class NPZFileHandler(BaseFileHandler):
    def __init__(self):
        # A helper function to unpack numpy 0-d arrays
        self._unpack = self._get_unpack_helper()

    def _get_unpack_helper(self):
        def unpack_helper(value: Any, default: Optional[Any] = None) -> Any:
            if value is None:
                return default
            if isinstance(value, np.ndarray) and value.shape == ():
                try:
                    return value.item()
                except Exception:
                    return default
            return value

        return unpack_helper

    def save(self, obj: Any, file_path: Path):
        file_path = file_path.with_suffix(".npz")

        try:
            np.savez_compressed(file_path, **obj)
        except Exception as e:
            raise IOError(f"Failed to save data to {file_path}: {e}") from e

    def load(self, file_path: Path) -> Dict[str, Any]:
        if not file_path.exists():
            raise FileNotFoundError(f"No data found at {file_path}")
        try:
            with np.load(file_path, allow_pickle=True) as data:
                return {key: self._unpack(data.get(key)) for key in data.files}
        except Exception as e:
            raise IOError(f"Failed to load data from {file_path}: {e}") from e
