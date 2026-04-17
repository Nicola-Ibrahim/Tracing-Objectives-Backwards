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


def main():
    log_header("Environment Initialization")
    root_dir = get_root_dir()

    # 1. Boot Infrastructure (Docker)
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

    # 2. Force Environment Sync
    # Note: init_env already has its own headers/logs
    # Now that Docker is up, init_env will be able to flush Redis successfully
    run_command(
        "python3 scripts/init_env.py",
        description="Initializing environment variables",
        cwd=root_dir,
    )

    # 3. Dependency Sync
    log_header("Dependency Synchronization")
    log_info("Ensuring that local packages are up to date with the latest changes.")

    # 3. Backend Dependencies
    if is_tool_installed("uv"):
        run_command(
            "uv sync",
            description="Updating backend dependencies (uv)",
            cwd=root_dir,
        )
    else:
        log_error("uv not found. Backend dependencies could not be synchronized.")

    log_header("Workspace Ready")
    log_success("All containers are running in the background.")
    log_info("Enjoy your coding session! ☕")


if __name__ == "__main__":
    main()
