import shutil

from ..engine.system import get_root_dir, is_tool_installed
from ..engine.unit_of_work import Command, UnitOfWork


def sync_dependencies(skip_confirm: bool = False) -> bool:
    """Ensure that local packages are up to date with the latest changes."""
    with UnitOfWork(
        "Dependency Synchronization",
        info="Ensuring that local packages are up to date with the latest changes.",
    ) as work:
        if not is_tool_installed("uv"):
            work.logger.error(
                "uv not found. Backend dependencies could not be synchronized."
            )
            return False

        cmd = Command(
            work,
            cmd="uv sync",
            description="Updating backend dependencies (uv)",
            skip_confirm=skip_confirm,
        )

        if not cmd.success:
            work.logger.error("Dependency sync failed. Try running 'uv sync' manually.")
            return False

    return True


def cleanup_storage(skip_confirm: bool = False) -> None:
    """Scrub local backend storage (uploads, logs, temporary files)."""
    storage_dir = get_root_dir() / "storage"
    with UnitOfWork(
        "Ephemeral Data Cleanup",
        info="Cleaning local backend storage (uploads, logs, temporary files).",
    ) as work:
        if not skip_confirm:
            if not work.logger.confirm("Scrub all files in storage directory?"):
                work.logger.info_step("Cleanup aborted by user.")
                return

        if storage_dir.exists():
            try:
                for item in storage_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                # Ensure it stays as a directory
                storage_dir.mkdir(exist_ok=True)
                work.logger.success("Backend storage is now clean.")
            except Exception as e:
                work.logger.error(f"Failed to scrub storage: {str(e)}")
        else:
            storage_dir.mkdir(parents=True, exist_ok=True)
            work.logger.success("Clean storage directory created.")


def cleanup_caches(skip_confirm: bool = False) -> None:
    """Remove temporary runtime artifacts and Python caches."""
    with UnitOfWork(
        "Cache Cleanup",
        info="Removing temporary runtime artifacts and Python caches.",
    ) as work:
        cleanup_cmd = (
            r'find . -type d \( -name "__pycache__" '
            r'-o -name ".pytest_cache" -o -name ".ruff_cache" \) '
            r"-exec rm -rf {} + 2>/dev/null || true"
        )
        Command(
            work,
            cmd=cleanup_cmd,
            description="Clearing Python & Linter caches",
            confirm=True,
            skip_confirm=skip_confirm,
        )


def audit_storage(skip_confirm: bool = False) -> None:
    """Report on the disk space used by AI datasets and artifacts."""
    with UnitOfWork(
        "Storage Audit",
    ) as work:
        storage_paths = []
        root = get_root_dir()
        if (root / "storage").exists():
            storage_paths.append("storage")
        if (root / "data").exists():
            storage_paths.append("data")

        if storage_paths:
            work.logger.info(
                "Calculating disk space used by AI datasets and artifacts."
            )
            audit_cmd = f"du -sh {' '.join(storage_paths)}"
            Command(
                work,
                cmd=audit_cmd,
                description="Auditing local storage size",
                confirm=True,
                skip_confirm=skip_confirm,
            )
        else:
            work.logger.info("No local storage/data directories found to audit.")


def summarize_work(skip_confirm: bool = False) -> None:
    """Provide a quick briefing on uncommitted changes."""
    with UnitOfWork(
        "Work Summary",
    ) as work:
        if is_tool_installed("git"):
            work.logger.info(
                "Checking for uncommitted changes before closing the session."
            )
            Command(
                work,
                cmd="git status -s",
                description="Listing uncommitted files",
                confirm=True,
                skip_confirm=skip_confirm,
            )
            work.logger.warning("Don't forget to commit your changes if you're done!")
        else:
            work.logger.error("Git not found. Could not generate workspace summary.")
