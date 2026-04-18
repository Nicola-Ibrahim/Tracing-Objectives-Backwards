from pathlib import Path

from .shell import (
    is_tool_installed,
    run_command,
)
from .ui import (
    log_error,
    log_header,
    log_info,
    log_warning,
)


def boot_infrastructure(root_dir: Path, skip_confirm: bool = False):
    """Launch all background services and application containers."""
    log_header("Infrastructure & App Services")
    log_info("Launching all background services and application containers.")

    if is_tool_installed("docker"):
        # Launching everything defined in docker-compose.yml
        base_cmd = "docker compose up -d --build"
        if is_tool_installed("doppler"):
            cmd = f"doppler run -- {base_cmd}"
        else:
            cmd = base_cmd
            log_warning("Doppler not found. Using local environment only.")

        run_command(
            cmd,
            description="Booting all Docker containers",
            cwd=root_dir,
            stream=True,
            skip_confirm=skip_confirm,
        )
    else:
        log_error("Docker daemon is not running. Application startup failed.")


def shutdown_services(
    root_dir: Path, confirm: bool = False, skip_confirm: bool = False
):
    """Gracefully stop all Docker services."""
    log_header("Shutdown Services")
    if is_tool_installed("docker"):
        down_cmd = "docker compose down"
        run_command(
            down_cmd,
            description="Stopping all Docker containers",
            cwd=root_dir,
            confirm=confirm,
            skip_confirm=skip_confirm,
        )
    else:
        log_error("Docker CLI not found. Could not stop containers.")


def reset_infrastructure(root_dir: Path, skip_confirm: bool = False):
    """Ensure all background infrastructure is in a clean state (e.g., Redis)."""
    log_header("Infrastructure Reset")
    log_info("Ensuring all background infrastructure is in a clean state.")

    if is_tool_installed("docker"):
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
    else:
        log_warning("Docker not installed. Skipping infrastructure reset.")
