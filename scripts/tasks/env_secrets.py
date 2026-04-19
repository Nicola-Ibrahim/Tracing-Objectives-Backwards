import subprocess

from ..engine.system import is_tool_installed
from ..engine.task_group import Task, TaskGroup
from ..engine.ui import Logger


def setup_identity(skip_confirm: bool = False) -> bool:
    """Handle authentication with platform services (Doppler)."""
    if not is_tool_installed("doppler"):
        Logger().error(
            "Doppler CLI not found in PATH. Please install it to sync secrets."
        )
        return False

    # Check if already logged in - if so, we don't need a box at all
    auth_check = subprocess.run(["doppler", "me"], capture_output=True, text=True)
    if auth_check.returncode == 0:
        return True

    with TaskGroup("Identity & Secrets") as group:
        group.logger.info("Launching Doppler interactive login flow.")
        task = Task(
            cmd="doppler login",
            description="Logging into Doppler CLI",
            interactive=True,
            skip_confirm=skip_confirm,
        )

        if not task.success:
            group.logger.error("Doppler login failed. Secrets synchronization skipped.")
            return False

    return True


def sync_secrets(skip_confirm: bool = False) -> bool:
    """Synchronize local secrets with the latest values from Doppler."""
    if not is_tool_installed("doppler"):
        Logger().error("Doppler CLI not found. Manual .env setup required.")
        return False

    # Check if logged in
    login_check = subprocess.run("doppler me", shell=True, capture_output=True)
    if login_check.returncode != 0:
        if not setup_identity(skip_confirm=skip_confirm):
            return False

    with TaskGroup(
        "Secret Management",
        info="Pulling latest environment variables (.env) from Doppler.",
    ):
        task = Task(
            cmd="doppler secrets download --format env --no-file > .env",
            description="Downloading .env file (Doppler)",
            skip_confirm=skip_confirm,
        )

        if not task.success:
            return False

    return True
