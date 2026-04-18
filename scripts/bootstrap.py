import platform

from .utils.env_secrets import setup_identity
from .utils.shell import (
    is_tool_installed,
    run_command,
)
from .utils.ui import (
    log_header,
    log_info,
    log_success,
    log_warning,
)


def check_installed_tools():
    """Verify if core developer tools are already available."""
    log_header("Checking Installed Tools")

    tools_to_check = {
        "uv": "uv --version",
        "doppler": "doppler --version",
    }

    missing_tools = []
    for tool in tools_to_check:
        if not is_tool_installed(tool):
            log_warning(f"{tool} is NOT installed.")
            missing_tools.append(tool)
        else:
            log_success(f"{tool} is available.")

    if not missing_tools:
        log_success("All core tools are ready to go.")

    return missing_tools


def install_missing_tools(missing_tools):
    """Orchestrate the installation of missing tools based on OS."""
    if not missing_tools:
        return

    os_type = platform.system()
    log_header(f"Installing missing tools: {', '.join(missing_tools)}")

    # Linux / macOS (Unix-like)
    if os_type in ["Linux", "Darwin"]:
        has_brew = is_tool_installed("brew")

        for tool in missing_tools:
            if tool == "uv":
                if os_type == "Darwin" and has_brew:
                    run_command(
                        "brew install uv", "Installing uv Python tool (Homebrew)"
                    )
                else:
                    run_command(
                        "curl -fsSL https://astral.sh/uv/install.sh | sh",
                        "Installing uv Python tool (curl)",
                    )

            elif tool == "doppler":
                if os_type == "Darwin" and has_brew:
                    run_command(
                        "brew install dopplerhq/cli/doppler",
                        "Installing Doppler CLI secret manager (Homebrew)",
                    )
                else:
                    log_header("Manual Setup Required")
                    log_info("Visit: https://docs.doppler.com/docs/install-cli")

    # Windows
    elif os_type == "Windows":
        has_winget = is_tool_installed("winget")

        for tool in missing_tools:
            if tool == "uv":
                if has_winget:
                    run_command(
                        "winget install uv", "Installing uv Python tool (winget)"
                    )
                else:
                    run_command(
                        'powershell -ExecutionPolicy ByPass -c "'
                        'irm https://astral.sh/uv/install.ps1 | iex"',
                        "Installing uv Python tool (PowerShell)",
                    )

            elif tool == "doppler":
                if has_winget:
                    run_command(
                        "winget install doppler",
                        "Installing Doppler secrets (winget)",
                    )
                else:
                    log_warning("Manual installation suggested for Doppler on Windows.")
                    log_info("Visit: https://docs.doppler.com/docs/install-cli")


def main():
    log_header("Environment Bootstrap")
    log_info("Ensuring all core developer tools are installed and configured.")

    # 1. Check for tools
    missing = check_installed_tools()

    # 2. Install if needed
    install_missing_tools(missing)

    # 3. Setup authentication (passing skip_confirm as False for initial bootstrap)
    setup_identity(skip_confirm=False)

    log_header("Environment Setup Complete")
    log_success("Bootstrap finished successfully!")
    log_info(
        "NEXT STEP: Run 'python3 -m scripts.dev up' to launch "
        "your daily development workspace."
    )


if __name__ == "__main__":
    main()
