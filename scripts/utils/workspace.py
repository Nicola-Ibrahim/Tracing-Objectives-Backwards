import shutil
from pathlib import Path

from .shell import is_tool_installed, run_command
from .ui import (
    log_error,
    log_header,
    log_info,
    log_success,
    log_warning,
)


def sync_dependencies(root_dir: Path, skip_confirm: bool = False) -> bool:
    """Ensure that local packages are up to date with the latest changes."""
    log_header("Dependency Synchronization")
    log_info("Ensuring that local packages are up to date with the latest changes.")

    if not is_tool_installed("uv"):
        log_error("uv not found. Backend dependencies could not be synchronized.")
        return False

    success = run_command(
        "uv sync",
        description="Updating backend dependencies (uv)",
        cwd=root_dir,
        skip_confirm=skip_confirm,
    )

    if not success:
        log_error("Dependency sync failed. Try running 'uv sync' manually.")
        return False

    return True


def cleanup_storage(root_dir: Path, skip_confirm: bool = False) -> None:
    """Scrub local backend storage (uploads, logs, temporary files)."""
    storage_dir = root_dir / "storage"
    log_header("Ephemeral Data Cleanup")
    log_info("Cleaning local backend storage (uploads, logs, temporary files).")

    if storage_dir.exists():
        try:
            for item in storage_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            # Ensure it stays as a directory
            storage_dir.mkdir(exist_ok=True)
            log_success("Backend storage is now clean.")
        except Exception as e:
            log_error(f"Failed to scrub storage: {str(e)}")
    else:
        storage_dir.mkdir(parents=True, exist_ok=True)
        log_success("Clean storage directory created.")


def cleanup_caches(
    root_dir: Path, confirm: bool = False, skip_confirm: bool = False
) -> None:
    """Remove temporary runtime artifacts and Python caches."""
    log_header("Cache Cleanup")
    log_info("Removing temporary runtime artifacts and Python caches.")

    cleanup_cmd = (
        r'find . -type d \( -name "__pycache__" '
        r'-o -name ".pytest_cache" -o -name ".ruff_cache" \) '
        r"-exec rm -rf {} + 2>/dev/null || true"
    )
    run_command(
        cleanup_cmd,
        description="Clearing Python & Linter caches",
        cwd=root_dir,
        confirm=confirm,
        skip_confirm=skip_confirm,
    )


def audit_storage(
    root_dir: Path, confirm: bool = False, skip_confirm: bool = False
) -> None:
    """Report on the disk space used by AI datasets and artifacts."""
    log_header("Storage Audit")
    storage_paths = []
    if (root_dir / "storage").exists():
        storage_paths.append("storage")
    if (root_dir / "data").exists():
        storage_paths.append("data")

    if storage_paths:
        log_info("Calculating disk space used by AI datasets and artifacts.")
        audit_cmd = f"du -sh {' '.join(storage_paths)}"
        run_command(
            audit_cmd,
            description="Auditing local storage size",
            cwd=root_dir,
            confirm=confirm,
            skip_confirm=skip_confirm,
        )
    else:
        log_info("No local storage/data directories found to audit.")


def summarize_work(
    root_dir: Path, confirm: bool = False, skip_confirm: bool = False
) -> None:
    """Provide a quick briefing on uncommitted changes."""
    log_header("Work Summary")
    if is_tool_installed("git"):
        log_info("Checking for uncommitted changes before closing the session.")
        run_command(
            "git status -s",
            description="Listing uncommitted files",
            cwd=root_dir,
            confirm=confirm,
            skip_confirm=skip_confirm,
        )
        log_warning("Don't forget to commit your changes if you're done!")
    else:
        log_error("Git not found. Could not generate workspace summary.")
