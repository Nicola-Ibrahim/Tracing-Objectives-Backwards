from utils import log_header, log_step, log_success, log_info, log_error, get_root_dir, is_tool_installed, launch_terminal_tab, run_command


def main():
    log_header("Morning Workspace Setup")
    log_info("Preparing all services, dependencies, and environments for development.")
    root_dir = get_root_dir()

    # 1. Sync Secrets & Environment State
    # Note: init_env already has its own headers/logs
    run_command("python3 scripts/init_env.py", description="Initializing environment variables", cwd=root_dir)

    # 2. Dependency Sync
    log_header("Dependency Synchronization")
    log_info("Ensuring that local packages are up to date with the latest changes.")
    if is_tool_installed("uv"):
        run_command("uv sync", description="Updating backend dependencies (uv)", cwd=root_dir / "backend")
    else:
        log_error("uv not found. Backend dependencies could not be synchronized.")

    if is_tool_installed("pnpm"):
        run_command("pnpm install", description="Updating frontend dependencies (pnpm)", cwd=root_dir / "frontend")
    else:
        log_error("pnpm not found. Frontend dependencies could not be synchronized.")

    # 3. Docker Containers
    log_header("Infrastructure Services")
    log_info("Launching background background services (Redis, Nginx).")
    if is_tool_installed("docker"):
        run_command("docker compose up -d redis nginx", description="Booting Docker containers", cwd=root_dir)
    else:
        log_error("Docker daemon is not running. Background services skipped.")

    # 4. Launch Servers
    log_header("Interactive Servers")
    log_info("Automating the launch of interactive development servers.")
    
    # Backend Server (FastAPI)
    backend_cmd = "uv run poe serve"
    launch_terminal_tab(backend_cmd, "FastAPI Backend", root_dir / "backend")

    # Frontend Server (Next.js)
    frontend_cmd = "pnpm dev"
    launch_terminal_tab(frontend_cmd, "Next.js Frontend", root_dir / "frontend")

    log_header("Workspace Ready")
    log_success("All servers are active and services are running.")
    log_info("Enjoy your coding session! ☕")


if __name__ == "__main__":
    main()
