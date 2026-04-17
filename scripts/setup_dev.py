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


def boot_infrastructure(root_dir: Path):
    """Launch all background services and application containers."""
    log_header("Infrastructure & App Services")
    log_info("Launching all background services and application containers.")
    
    if is_tool_installed("docker"):
        # Launching everything defined in docker-compose.yml
        # Note: using doppler run -- to inject secrets at build and runtime
        base_cmd = "docker compose up -d --build"
        if is_tool_installed("doppler"):
            cmd = f"doppler run -- {base_cmd}"
        else:
            cmd = base_cmd
            log_warning("Doppler not found. Using local environment only.")

        run_command(
            cmd, description="Booting all Docker containers", cwd=root_dir, stream=True
        )
    else:
        log_error("Docker daemon is not running. Application startup failed.")


def sync_environment(root_dir: Path):
    """Synchronize environment variables and clean state."""
    # Note: init_env already has its own internal headers/logs
    run_command(
        "python3 scripts/init_env.py",
        description="Initializing environment variables",
        cwd=root_dir,
    )


def sync_dependencies(root_dir: Path):
    """Ensure that local packages are up to date with the latest changes."""
    log_header("Dependency Synchronization")
    log_info("Ensuring that local packages are up to date with the latest changes.")

    if is_tool_installed("uv"):
        run_command(
            "uv sync",
            description="Updating backend dependencies (uv)",
            cwd=root_dir,
        )
    else:
        log_error("uv not found. Backend dependencies could not be synchronized.")


def main():
    log_header("Development Setup")
    root_dir = get_root_dir()

    # 1. Boot Infrastructure
    boot_infrastructure(root_dir)

    # 2. Sync Environment
    sync_environment(root_dir)

    # 3. Sync Dependencies
    sync_dependencies(root_dir)

    log_header("Workspace Ready")
    log_success("All containers are running in the background.")
    log_info("Enjoy your coding session! ☕")


if __name__ == "__main__":
    main()
