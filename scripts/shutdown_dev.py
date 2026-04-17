from pathlib import Path

from utils import (
    get_root_dir,
    is_tool_installed,
    log_error,
    log_header,
    log_info,
    log_success,
    log_warning,
    run_command,
)


def shutdown_services(root_dir: Path):
    """Gracefully stop all Docker services."""
    if is_tool_installed("docker"):
        down_cmd = "docker compose down"
        run_command(
            down_cmd,
            description="Stopping all Docker containers",
            cwd=root_dir,
            confirm=True,
        )
    else:
        log_error("Docker CLI not found. Could not stop containers.")


def cleanup_caches(root_dir: Path):
    """Remove temporary runtime artifacts and Python caches."""
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
        confirm=True,
    )


def audit_storage(root_dir: Path):
    """Report on the disk space used by AI datasets and artifacts."""
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
            confirm=True,
        )
    else:
        log_info("No local storage/data directories found to audit.")


def summarize_work(root_dir: Path):
    """Provide a quick briefing on uncommitted changes."""
    if is_tool_installed("git"):
        log_info("Checking for uncommitted changes before closing the session.")
        run_command(
            "git status -s",
            description="Listing uncommitted files",
            cwd=root_dir,
            confirm=True,
        )
        log_warning("Don't forget to commit your changes if you're done!")
    else:
        log_error("Git not found. Could not generate workspace summary.")


def main():
    root_dir = get_root_dir()

    log_header("Session Termination")
    log_info("Gracefully shutting down services and cleaning up session data.")

    # Orchestrate shutdown steps using declarative decoration
    shutdown_services(root_dir)
    cleanup_caches(root_dir)
    audit_storage(root_dir)
    summarize_work(root_dir)

    log_header("Shutdown Complete")
    log_success("All background services have been terminated.")
    log_info("See you tomorrow! 🌙")


if __name__ == "__main__":
    main()
