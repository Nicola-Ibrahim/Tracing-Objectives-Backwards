import shutil
from pathlib import Path

from .system import is_tool_installed
from .jobs import Job, Command


def sync_dependencies(root_dir: Path, skip_confirm: bool = False) -> bool:
    """Ensure that local packages are up to date with the latest changes."""
    with Job(
        "Dependency Synchronization",
        info="Ensuring that local packages are up to date with the latest changes.",
    ) as job:
        if not is_tool_installed("uv"):
            job.logger.error("uv not found. Backend dependencies could not be synchronized.")
            return False

        cmd = Command(
            job.logger,
            cmd="uv sync",
            description="Updating backend dependencies (uv)",
            cwd=root_dir,
            skip_confirm=skip_confirm,
        )

        if not cmd.success:
            job.logger.error("Dependency sync failed. Try running 'uv sync' manually.")
            return False

    return True


def cleanup_storage(root_dir: Path, skip_confirm: bool = False) -> None:
    """Scrub local backend storage (uploads, logs, temporary files)."""
    storage_dir = root_dir / "storage"
    with Job(
        "Ephemeral Data Cleanup",
        info="Cleaning local backend storage (uploads, logs, temporary files).",
        confirm=False,
    ) as job:
        if storage_dir.exists():
            try:
                for item in storage_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                # Ensure it stays as a directory
                storage_dir.mkdir(exist_ok=True)
                job.logger.success("Backend storage is now clean.")
            except Exception as e:
                job.logger.error(f"Failed to scrub storage: {str(e)}")
        else:
            storage_dir.mkdir(parents=True, exist_ok=True)
            job.logger.success("Clean storage directory created.")


def cleanup_caches(
    root_dir: Path, confirm: bool = False, skip_confirm: bool = False
) -> None:
    """Remove temporary runtime artifacts and Python caches."""
    with Job(
        "Cache Cleanup",
        info="Removing temporary runtime artifacts and Python caches.",
        confirm=confirm,
    ) as job:
        cleanup_cmd = (
            r'find . -type d \( -name "__pycache__" '
            r'-o -name ".pytest_cache" -o -name ".ruff_cache" \) '
            r"-exec rm -rf {} + 2>/dev/null || true"
        )
        Command(
            job.logger,
            cmd=cleanup_cmd,
            description="Clearing Python & Linter caches",
            cwd=root_dir,
            skip_confirm=skip_confirm,
        )


def audit_storage(
    root_dir: Path, confirm: bool = False, skip_confirm: bool = False
) -> None:
    """Report on the disk space used by AI datasets and artifacts."""
    with Job(
        "Storage Audit",
        confirm=confirm,
    ) as job:
        storage_paths = []
        if (root_dir / "storage").exists():
            storage_paths.append("storage")
        if (root_dir / "data").exists():
            storage_paths.append("data")

        if storage_paths:
            job.logger.info("Calculating disk space used by AI datasets and artifacts.")
            audit_cmd = f"du -sh {' '.join(storage_paths)}"
            Command(
                job.logger,
                cmd=audit_cmd,
                description="Auditing local storage size",
                cwd=root_dir,
                skip_confirm=skip_confirm,
            )
        else:
            job.logger.info("No local storage/data directories found to audit.")


def summarize_work(
    root_dir: Path, confirm: bool = False, skip_confirm: bool = False
) -> None:
    """Provide a quick briefing on uncommitted changes."""
    with Job(
        "Work Summary",
        confirm=confirm,
    ) as job:
        if is_tool_installed("git"):
            job.logger.info("Checking for uncommitted changes before closing the session.")
            Command(
                job.logger,
                cmd="git status -s",
                description="Listing uncommitted files",
                cwd=root_dir,
                skip_confirm=skip_confirm,
            )
            job.logger.warning("Don't forget to commit your changes if you're done!")
        else:
            job.logger.error("Git not found. Could not generate workspace summary.")
