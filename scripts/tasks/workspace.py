import shutil

from ..engine.system import get_root_dir, is_tool_installed
from ..engine.task_group import Task, TaskGroup
from ..engine.ui import Logger


def sync_dependencies(skip_confirm: bool = False) -> bool:
    """Ensure that local packages are up to date with the latest changes."""
    if not is_tool_installed("uv"):
        Logger().error("uv not found. Backend dependencies could not be synchronized.")
        return False

    with TaskGroup(
        "Dependency Synchronization",
        info="Ensuring that local packages are up to date with the latest changes.",
    ):
        task = Task(
            cmd="uv sync",
            description="Updating backend dependencies (uv)",
            skip_confirm=skip_confirm,
        )

        if not task.success:
            return False

    return True


def cleanup_storage(skip_confirm: bool = False) -> None:
    """Scrub local backend storage (uploads, logs, temporary files)."""
    storage_dir = get_root_dir() / "storage"

    # If it doesn't exist, we'll just create it without a big box
    if not storage_dir.exists():
        storage_dir.mkdir(parents=True, exist_ok=True)
        return

    with TaskGroup(
        "Ephemeral Data Cleanup",
        info="Cleaning local backend storage (uploads, logs, temporary files).",
    ) as group:
        if not skip_confirm:
            if not group.logger.confirm("Scrub all files in storage directory?"):
                group.logger.info_step("Cleanup aborted by user.")
                return

        try:
            for item in storage_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            # Ensure it stays as a directory
            storage_dir.mkdir(exist_ok=True)
            group.logger.success("Backend storage is now clean.")
        except Exception as e:
            group.logger.error(f"Failed to scrub storage: {str(e)}")


def cleanup_caches(skip_confirm: bool = False) -> None:
    """Remove temporary runtime artifacts and Python caches."""
    with TaskGroup(
        "Cache Cleanup",
        info="Removing temporary runtime artifacts and Python caches.",
    ):
        cleanup_cmd = (
            r'find . -type d \( -name "__pycache__" '
            r'-o -name ".pytest_cache" -o -name ".ruff_cache" \) '
            r"-exec rm -rf {} + 2>/dev/null || true"
        )
        Task(
            cmd=cleanup_cmd,
            description="Clearing Python & Linter caches",
            confirm=True,
            skip_confirm=skip_confirm,
        )


def audit_storage(skip_confirm: bool = False) -> None:
    """Report on the disk space used by AI datasets and artifacts."""
    storage_paths = []
    root = get_root_dir()
    if (root / "storage").exists():
        storage_paths.append("storage")
    if (root / "data").exists():
        storage_paths.append("data")

    if not storage_paths:
        return

    with TaskGroup(
        "Storage Audit",
    ) as group:
        group.logger.info("Calculating disk space used by AI datasets and artifacts.")
        audit_cmd = f"du -sh {' '.join(storage_paths)}"
        Task(
            cmd=audit_cmd,
            description="Auditing local storage size",
            confirm=True,
            skip_confirm=skip_confirm,
        )


def summarize_work(skip_confirm: bool = False) -> None:
    """Provide a quick briefing on uncommitted changes."""
    if not is_tool_installed("git"):
        return

    with TaskGroup(
        "Work Summary",
    ) as group:
        group.logger.info(
            "Checking for uncommitted changes before closing the session."
        )
        Task(
            cmd="git status -s",
            description="Listing uncommitted files",
            confirm=True,
            skip_confirm=skip_confirm,
        )
        group.logger.warning("Don't forget to commit your changes if you're done!")
