import subprocess
from pathlib import Path

from .shell import is_tool_installed, run_command
from .ui import logger


def setup_identity(skip_confirm: bool = False) -> bool:
    """Handle authentication with platform services (Doppler)."""
    logger.header("Identity & Secrets")

    if not is_tool_installed("doppler"):
        logger.error(
            "Doppler CLI not found in PATH. Please install it to sync secrets."
        )
        return False

    # Check if already logged in
    auth_check = subprocess.run(["doppler", "me"], capture_output=True, text=True)
    if auth_check.returncode == 0:
        logger.success("Already authenticated with Doppler CLI.")
        return True

    logger.info("Launching Doppler interactive login flow.")
    success = run_command(
        "doppler login",
        "Logging into Doppler CLI",
        interactive=True,
        skip_confirm=skip_confirm,
    )

    if not success:
        logger.error("Doppler login failed. Secrets synchronization skipped.")
        return False

    return True


def sync_secrets(root_dir: Path, skip_confirm: bool = False) -> bool:
    """Synchronize local secrets with the latest values from Doppler."""
    logger.header("Secret Management")
    logger.info("Pulling latest environment variables (.env) from Doppler.")

    if not is_tool_installed("doppler"):
        logger.error("Doppler CLI not found. Manual .env setup required.")
        return False

    # Check if logged in
    login_check = subprocess.run("doppler me", shell=True, capture_output=True)
    if login_check.returncode != 0:
        logger.warning("Not authenticated. Launching login...")
        if not setup_identity(skip_confirm=skip_confirm):
            return False

    # Pull secrets to .env
    success = run_command(
        "doppler secrets download --format env --no-file > .env",
        description="Downloading .env file (Doppler)",
        cwd=root_dir,
        skip_confirm=skip_confirm,
    )

    if not success:
        logger.error("Failed to sync secrets from Doppler. Environment may be stale.")
        return False

    logger.success("Secrets successfully synchronized.")
    return True
