import shutil
import subprocess

from utils import (
    get_root_dir,
    is_tool_installed,
    log_error,
    log_header,
    log_info,
    log_step,
    log_success,
    log_warning,
    run_command,
)


def main():
    log_header("Environment Initialization")
    log_info("Synchronizing local environment state with global settings.")
    root_dir = get_root_dir()

    # 1. Secret Management (Doppler)
    log_header("Secret Management")
    log_info("Pulling latest environment variables (.env) from Doppler.")
    if is_tool_installed("doppler"):
        # Check if logged in
        login_check = subprocess.run("doppler me", shell=True, capture_output=True)
        if login_check.returncode != 0:
            log_warning("Not authenticated. Launching login...")
            run_command("doppler login", "Authenticating Doppler", interactive=True)

        # Pull secrets to .env
        run_command(
            "doppler secrets download --format env --no-file > .env",
            description="Downloading .env file (Doppler)",
            cwd=root_dir,
        )
        log_success("Secrets successfully synchronized.")
    else:
        log_error("Doppler CLI not found. Manual .env setup required.")

    # 2. Redis Reset
    log_header("Infrastructure Reset")
    log_info("Ensuring all background infrastructure is in a clean state.")
    try:
        # Try via docker-compose first
        run_command(
            "docker compose exec -T redis redis-cli flushall",
            description="Flushing Redis cache (Docker)",
            cwd=root_dir,
            exit_on_error=False,
        )
    except Exception:
        log_warning("Could not flush Redis. It might not be running.")

    # 3. Storage Cleanup
    storage_dir = root_dir / "storage"
    log_header("Ephemeral Data Cleanup")
    log_info("Cleaning local backend storage (uploads, logs, temporary files).")

    if storage_dir.exists():
        log_step(f"Scrubbing database and temporary files in {storage_dir}")
        try:
            for item in storage_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            # Ensure it stays as a directory
            storage_dir.mkdir(exist_ok=True)
            log_success("Backend storage is now clean.")
        except Exception as e:
            log_error(f"Failed to scrub storage: {str(e)}")
    else:
        storage_dir.mkdir(parents=True, exist_ok=True)
        log_success("Clean storage directory created.")

    log_header("Initialization Complete")
    log_success("Your environment is now synchronized and clean.")
    log_info(
        "Ready to work? Run 'python3 scripts/setup_dev.py' to launch your morning workspace."
    )


if __name__ == "__main__":
    main()
