import subprocess
from pathlib import Path

from .shell import is_tool_installed, run_command
from .ui import (
    log_error,
    log_header,
    log_info,
    log_success,
    log_warning,
)


def setup_identity(skip_confirm: bool = False) -> bool:
    """Handle authentication with platform services (Doppler)."""
    log_header("Identity & Secrets")

    if not is_tool_installed("doppler"):
        log_error("Doppler CLI not found in PATH. Please install it to sync secrets.")
        return False

    # Check if already logged in
    auth_check = subprocess.run(["doppler", "me"], capture_output=True, text=True)
    if auth_check.returncode == 0:
        log_success("Already authenticated with Doppler CLI.")
        return True

    log_info("Launching Doppler interactive login flow.")
    success = run_command(
        "doppler login",
        "Logging into Doppler CLI",
        interactive=True,
        skip_confirm=skip_confirm,
    )

    if not success:
        log_error("Doppler login failed. Secrets synchronization skipped.")
        return False

    return True


def sync_secrets(root_dir: Path, skip_confirm: bool = False) -> bool:
    """Synchronize local secrets with the latest values from Doppler."""
    log_header("Secret Management")
    log_info("Pulling latest environment variables (.env) from Doppler.")

    if not is_tool_installed("doppler"):
        log_error("Doppler CLI not found. Manual .env setup required.")
        return False

    # Check if logged in
    login_check = subprocess.run("doppler me", shell=True, capture_output=True)
    if login_check.returncode != 0:
        log_warning("Not authenticated. Launching login...")
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
        log_error("Failed to sync secrets from Doppler. Environment may be stale.")
        return False

    log_success("Secrets successfully synchronized.")
    return True
