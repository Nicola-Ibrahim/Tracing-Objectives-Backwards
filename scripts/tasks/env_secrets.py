import subprocess

from ..engine.system import is_tool_installed
from ..engine.unit_of_work import Command, UnitOfWork


def setup_identity(skip_confirm: bool = False) -> bool:
    """Handle authentication with platform services (Doppler)."""
    with UnitOfWork("Identity & Secrets") as work:
        if not is_tool_installed("doppler"):
            work.logger.error(
                "Doppler CLI not found in PATH. Please install it to sync secrets."
            )
            return False

        # Check if already logged in
        auth_check = subprocess.run(["doppler", "me"], capture_output=True, text=True)
        if auth_check.returncode == 0:
            work.logger.success("Already authenticated with Doppler CLI.")
            return True

        work.logger.info("Launching Doppler interactive login flow.")
        cmd = Command(
            work,
            cmd="doppler login",
            description="Logging into Doppler CLI",
            interactive=True,
            skip_confirm=skip_confirm,
        )

        if not cmd.success:
            work.logger.error("Doppler login failed. Secrets synchronization skipped.")
            return False

    return True


def sync_secrets(skip_confirm: bool = False) -> bool:
    """Synchronize local secrets with the latest values from Doppler."""
    with UnitOfWork(
        "Secret Management",
        info="Pulling latest environment variables (.env) from Doppler.",
    ) as work:
        if not is_tool_installed("doppler"):
            work.logger.error("Doppler CLI not found. Manual .env setup required.")
            return False

        # Check if logged in
        login_check = subprocess.run("doppler me", shell=True, capture_output=True)
        if login_check.returncode != 0:
            work.logger.warning("Not authenticated. Launching login...")
            if not setup_identity(skip_confirm=skip_confirm):
                return False

        cmd = Command(
            work,
            cmd="doppler secrets download --format env --no-file > .env",
            description="Downloading .env file (Doppler)",
            skip_confirm=skip_confirm,
        )

        if not cmd.success:
            work.logger.error(
                "Failed to sync secrets from Doppler. Environment may be stale."
            )
            return False

    return True
