import argparse

from utils import (
    DIM,
    RESET,
    confirm_action,
    get_root_dir,
    is_tool_installed,
    log_error,
    log_header,
    log_info,
    log_step,
    log_success,
    run_command,
)


def main():
    parser = argparse.ArgumentParser(
        description="Terminate the development environment session."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe all Docker volumes and deep clean caches.",
    )
    args = parser.parse_args()

    if args.clean:
        if not confirm_action(
            "DEEP CLEAN requested. This will wipe Docker volumes and all build caches. Continue?"
        ):
            print(
                f"{DIM}Deep clean cancelled. Proceeding with standard shutdown...{RESET}"
            )
            args.clean = False

    log_header("Session Termination")
    log_info("Gracefully shutting down services and cleaning up session data.")
    root_dir = get_root_dir()

    # 1. Stop Docker Containers
    log_header("Service Shutdown")
    if is_tool_installed("docker"):
        down_cmd = "docker compose down"
        if args.clean:
            down_cmd += " --volumes --remove-orphans"
            log_info("Deep clean requested: Volumes will be removed.")

        run_command(
            down_cmd,
            description="Stopping all Docker containers",
            cwd=root_dir,
            confirm=True,
        )
    else:
        log_error("Docker CLI not found. Could not stop containers.")

    # 2. Cache Cleanup
    log_header("Cache Cleanup")
    log_info("Removing temporary runtime artifacts and Python caches.")

    # We use a shell command to find and remove caches to be thorough
    cleanup_cmd = (
        'find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true'
    )
    run_command(
        cleanup_cmd,
        description="Clearing Python caches (__pycache__)",
        cwd=root_dir,
        confirm=True,
    )

    if args.clean:
        deep_cleanup = "rm -rf .pytest_cache .ruff_cache .venv/src 2>/dev/null || true"
        run_command(
            deep_cleanup,
            description="Performing deep cache cleanup",
            cwd=root_dir,
            confirm=True,
        )

    log_header("Shutdown Complete")
    log_success("All background services have been terminated.")
    log_info("See you tomorrow! 🌙")


if __name__ == "__main__":
    main()
