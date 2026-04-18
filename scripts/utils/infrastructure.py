from pathlib import Path

from .jobs import Command, Job
from .system import is_tool_installed


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
    with Job(
        "Infrastructure & App Services",
        info="Launching all background services and application containers.",
        confirm=False,
    ) as job:
        if not is_tool_installed("docker"):
            job.logger.error("Docker CLI not found. Please install Docker Desktop.")
            return False

        if not _is_docker_daemon_running():
            job.logger.error("Docker daemon is not running. Please start Docker Desktop.")
            return False

        # Launching everything defined in docker-compose.yml
        base_cmd = "docker compose up -d --build"
        if is_tool_installed("doppler"):
            cmd = f"doppler run -- {base_cmd}"
        else:
            cmd = base_cmd
            job.logger.warning("Doppler not found. Using local environment only.")

        Command(
            job.logger,
            cmd=cmd,
            description="Booting all Docker containers",
            cwd=root_dir,
            stream=True,
            skip_confirm=skip_confirm,
        )
        return True


def shutdown_services(
    root_dir: Path, confirm: bool = False, skip_confirm: bool = False
) -> bool:
    """Gracefully stop all Docker services."""
    with Job(
        "Docker Shutdown",
        info="Gracefully stopping all Docker services.",
        confirm=confirm,
    ) as job:
        if not is_tool_installed("docker"):
            job.logger.warning("Docker CLI not found. Skipping service shutdown.")
            return True

        if not _is_docker_daemon_running():
            job.logger.warning("Docker daemon is not running. Could not stop containers.")
            return True

        Command(
            job.logger,
            cmd="docker compose down",
            description="Stopping all Docker containers",
            cwd=root_dir,
            skip_confirm=skip_confirm,
        )
        return True


def reset_infrastructure(root_dir: Path, skip_confirm: bool = False) -> bool:
    """Ensure all background infrastructure is in a clean state (e.g., Redis)."""
    with Job(
        "Infrastructure Reset",
        info="Ensuring all background infrastructure is in a clean state.",
        confirm=False,
    ) as job:
        if not is_tool_installed("docker") or not _is_docker_daemon_running():
            job.logger.warning("Docker unavailable. Skipping infrastructure reset.")
            return True

        try:
            # Try via docker-compose first
            Command(
                job.logger,
                cmd="docker compose exec -T redis redis-cli flushall",
                description="Flushing Redis cache (Docker)",
                cwd=root_dir,
                exit_on_error=False,
                skip_confirm=skip_confirm,
            )
            return True
        except Exception:
            job.logger.warning("Could not flush Redis. It might not be running.")
            return True
