from pathlib import Path
from typing import Protocol


class BaseResultArchiver(Protocol):
    def save(self, data: dict) -> Path: ...
