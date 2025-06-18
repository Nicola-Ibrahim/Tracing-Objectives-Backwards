from typing import Any

import joblib


class FileSystemSerializer:
    def save_model(self, model: Any, path: str):
        joblib.dump(model, path)
        print(f"Model artifact saved to {path}")

    def load_model(self, path: str) -> Any:
        return joblib.load(path)

    def save_metadata(self, path: str, metadata: dict[str, Any]):
        import json

        with open(path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved to {path}")

    def load_metadata(self, path: str) -> dict[str, Any]:
        import json

        with open(path, "r") as f:
            return json.load(f)
