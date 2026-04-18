from ..engine.system import is_tool_installed
from ..engine.unit_of_work import Command, UnitOfWork


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


def boot_infrastructure(skip_confirm: bool = False) -> bool:
    """Launch all background services and application containers."""
    with UnitOfWork(
        "Infrastructure & App Services",
        info="Launching all background services and application containers.",
    ) as work:
        if not is_tool_installed("docker"):
            work.logger.error("Docker CLI not found. Please install Docker Desktop.")
            return False

        if not _is_docker_daemon_running():
            work.logger.error(
                "Docker daemon is not running. Please start Docker Desktop."
            )
            return False

        # Launching everything defined in docker-compose.yml
        base_cmd = "docker compose up -d --build"
        if is_tool_installed("doppler"):
            cmd = f"doppler run -- {base_cmd}"
        else:
            cmd = base_cmd
            work.logger.warning("Doppler not found. Using local environment only.")

        Command(
            work,
            cmd=cmd,
            description="Booting all Docker containers",
            stream=True,
            skip_confirm=skip_confirm,
        )
        return True


def shutdown_services(skip_confirm: bool = False) -> bool:
    """Gracefully stop all Docker services."""
    with UnitOfWork(
        "Docker Shutdown",
        info="Gracefully stopping all Docker services.",
    ) as work:
        if not is_tool_installed("docker"):
            work.logger.warning("Docker CLI not found. Skipping service shutdown.")
            return True

        if not _is_docker_daemon_running():
            work.logger.warning(
                "Docker daemon is not running. Could not stop containers."
            )
            return True

        Command(
            work,
            cmd="docker compose down",
            description="Stopping all Docker containers",
            confirm=True,
            skip_confirm=skip_confirm,
        )
        return True


def reset_infrastructure(skip_confirm: bool = False) -> bool:
    """Ensure all background infrastructure is in a clean state (e.g., Redis)."""
    with UnitOfWork(
        "Infrastructure Reset",
        info="Ensuring all background infrastructure is in a clean state.",
    ) as work:
        if not is_tool_installed("docker") or not _is_docker_daemon_running():
            work.logger.warning("Docker unavailable. Skipping infrastructure reset.")
            return True

        try:
            # Try via docker-compose first
            Command(
                work,
                cmd="docker compose exec -T redis redis-cli flushall",
                description="Flushing Redis cache (Docker)",
                confirm=True,
                exit_on_error=False,
                skip_confirm=skip_confirm,
            )
            return True
        except Exception:
            work.logger.warning("Could not flush Redis. It might not be running.")
            return True
