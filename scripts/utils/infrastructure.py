from pathlib import Path

from .shell import is_tool_installed, run_command
from .ui import (
    log_error,
    log_header,
    log_info,
    log_warning,
)


def _is_docker_daemon_running() -> bool:
    """Verify if the Docker daemon is reachable."""
    try:
        # 'docker info' returns 0 if daemon is running
        import subprocess

        result = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, check=False
        )
        return result.returncode == 0
    except Exception:
        return False


def boot_infrastructure(root_dir: Path, skip_confirm: bool = False) -> bool:
    """Launch all background services and application containers."""
    log_header("Infrastructure & App Services")
    log_info("Launching all background services and application containers.")

    if not is_tool_installed("docker"):
        log_error("Docker CLI not found. Please install Docker Desktop.")
        return False

    if not _is_docker_daemon_running():
        log_error("Docker daemon is not running. Please start Docker Desktop.")
        return False

    # Launching everything defined in docker-compose.yml
    base_cmd = "docker compose up -d --build"
    if is_tool_installed("doppler"):
        cmd = f"doppler run -- {base_cmd}"
    else:
        cmd = base_cmd
        log_warning("Doppler not found. Using local environment only.")

    success = run_command(
        cmd,
        description="Booting all Docker containers",
        cwd=root_dir,
        stream=True,
        skip_confirm=skip_confirm,
    )

    if not success:
        log_error("Failed to boot Docker services. Check 'docker compose ps' for logs.")
        return False

    return True


def shutdown_services(
    root_dir: Path, confirm: bool = False, skip_confirm: bool = False
) -> bool:
    """Gracefully stop all Docker services."""
    log_header("Shutdown Services")

    if not is_tool_installed("docker"):
        log_warning("Docker CLI not found. Skipping service shutdown.")
        return True

    if not _is_docker_daemon_running():
        log_warning("Docker daemon is not running. Could not stop containers.")
        return True

    down_cmd = "docker compose down"
    success = run_command(
        down_cmd,
        description="Stopping all Docker containers",
        cwd=root_dir,
        confirm=confirm,
        skip_confirm=skip_confirm,
    )

    if not success:
        log_error("Docker stop command failed. You may need to kill containers manually.")
        return False

    return True


def reset_infrastructure(root_dir: Path, skip_confirm: bool = False) -> bool:
    """Ensure all background infrastructure is in a clean state (e.g., Redis)."""
    log_header("Infrastructure Reset")

    if not is_tool_installed("docker") or not _is_docker_daemon_running():
        log_warning("Docker unavailable. Skipping infrastructure reset.")
        return True

    log_info("Ensuring all background infrastructure is in a clean state.")
    try:
        # Try via docker-compose first
        run_command(
            "docker compose exec -T redis redis-cli flushall",
            description="Flushing Redis cache (Docker)",
            cwd=root_dir,
            exit_on_error=False,
            skip_confirm=skip_confirm,
        )
    except Exception:
        log_warning("Could not flush Redis. It might not be running.")

    return True
