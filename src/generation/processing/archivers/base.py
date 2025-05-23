from pathlib import Path
from typing import Protocol


class ResultArchiver(Protocol):
    def save(self, data: dict) -> Path: ...
