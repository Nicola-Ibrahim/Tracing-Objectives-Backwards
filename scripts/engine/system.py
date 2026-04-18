import shutil
from pathlib import Path


def get_root_dir():
    """Returns the absolute project root directory."""
    # Current structure: project/scripts/utils/system.py
    return Path(__file__).parent.parent.parent.absolute()


def is_tool_installed(name):
    """Checks if a tool is available in the system PATH."""
    return shutil.which(name) is not None
