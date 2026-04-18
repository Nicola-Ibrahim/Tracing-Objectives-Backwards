import platform

from .utils.env_secrets import setup_identity
from .utils.system import (
    is_tool_installed,
)
from .utils.jobs import Job, Command
from .utils.ui import Logger


def check_installed_tools() -> list[str]:
    """Verify if core developer tools are already available."""
    logger = Logger()
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
    with Job(
        f"Tool Installation: {', '.join(missing_tools)}",
        info=f"Orchestrating the installation of missing tools for {os_type}.",
    ) as job:
        # Linux / macOS (Unix-like)
        if os_type in ["Linux", "Darwin"]:
            has_brew = is_tool_installed("brew")

            for tool in missing_tools:
                if tool == "uv":
                    if os_type == "Darwin" and has_brew:
                        Command(
                            job.logger,
                            cmd="brew install uv",
                            description="Installing uv Python tool (Homebrew)",
                        )
                    else:
                        Command(
                            job.logger,
                            cmd="curl -fsSL https://astral.sh/uv/install.sh | sh",
                            description="Installing uv Python tool (curl)",
                        )

                elif tool == "doppler":
                    if os_type == "Darwin" and has_brew:
                        Command(
                            job.logger,
                            cmd="brew install dopplerhq/cli/doppler",
                            description="Installing Doppler CLI secret manager (Homebrew)",
                        )
                    else:
                        job.logger.header("Manual Setup Required")
                        job.logger.info("Visit: https://docs.doppler.com/docs/install-cli")

        # Windows
        elif os_type == "Windows":
            has_winget = is_tool_installed("winget")

            for tool in missing_tools:
                if tool == "uv":
                    if has_winget:
                        Command(
                            job.logger,
                            cmd="winget install uv",
                            description="Installing uv Python tool (winget)",
                        )
                    else:
                        Command(
                            job.logger,
                            cmd='powershell -ExecutionPolicy ByPass -c "'
                            'irm https://astral.sh/uv/install.ps1 | iex"',
                            description="Installing uv Python tool (PowerShell)",
                        )

                elif tool == "doppler":
                    if has_winget:
                        Command(
                            job.logger,
                            cmd="winget install doppler",
                            description="Installing Doppler secrets (winget)",
                        )
                    else:
                        job.logger.warning(
                            "Manual installation suggested for Doppler on Windows."
                        )
                        job.logger.info("Visit: https://docs.doppler.com/docs/install-cli")


def main() -> bool:
    """Entry point for the environment bootstrap process."""
    logger = Logger()
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
