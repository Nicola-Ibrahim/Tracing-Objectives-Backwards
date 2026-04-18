import platform

from .utils.env_secrets import setup_identity
from .utils.shell import (
    is_tool_installed,
    run_command,
)
from .utils.ui import logger


def check_installed_tools() -> list[str]:
    """Verify if core developer tools are already available."""
    logger.header("Checking Installed Tools")

    tools_to_check = {
        "uv": "uv --version",
        "doppler": "doppler --version",
    }

    missing_tools: list[str] = []
    for tool in tools_to_check:
        if not is_tool_installed(tool):
            logger.warning(f"{tool} is NOT installed.")
            missing_tools.append(tool)
        else:
            logger.success(f"{tool} is available.")

    if not missing_tools:
        logger.success("All core tools are ready to go.")

    return missing_tools


def install_missing_tools(missing_tools: list[str]) -> None:
    """Orchestrate the installation of missing tools based on OS."""
    if not missing_tools:
        return

    os_type = platform.system()
    logger.header(f"Installing missing tools: {', '.join(missing_tools)}")

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
                    logger.header("Manual Setup Required")
                    logger.info("Visit: https://docs.doppler.com/docs/install-cli")

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
                    logger.warning(
                        "Manual installation suggested for Doppler on Windows."
                    )
                    logger.info("Visit: https://docs.doppler.com/docs/install-cli")


def main() -> bool:
    """Entry point for the environment bootstrap process."""
    logger.header("Environment Bootstrap")
    logger.info("Ensuring all core developer tools are installed and configured.")

    # 1. Check for tools
    missing = check_installed_tools()

    # 2. Install if needed
    install_missing_tools(missing)

    # 3. Setup authentication (passing skip_confirm as False for initial bootstrap)
    if not setup_identity(skip_confirm=False):
        logger.error("Initial identity setup failed.")
        return False

    logger.header("Environment Setup Complete")
    logger.success("Bootstrap finished successfully!")
    logger.info(
        "NEXT STEP: Run 'python3 -m scripts.dev up' to launch "
        "your daily development workspace."
    )
    return True


if __name__ == "__main__":
    main()
