import pickle
from pathlib import Path

from .base import ResultArchiver


class PickleArchiver(ResultArchiver):
    """
    Archiver that saves the data in a pickle file.
    """

    def __init__(self, output_dir: Path = Path("data/raw")):
        self.output_dir = output_dir

    def save(self, data: dict) -> Path:
        self.output_dir.mkdir(exist_ok=True, parents=True)
        file_path = self.output_dir / f"pareto_{int(data['distance_km'])}km.pkl"
        with file_path.open("wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return file_path
