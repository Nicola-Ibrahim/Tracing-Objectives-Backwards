import platform

from ..engine.system import is_tool_installed
from ..engine.unit_of_work import Command, UnitOfWork


def check_core_tools() -> list[str]:
    """
    Verify if core developer tools (uv, doppler) are already available.
    Returns a list of missing tool names.
    """
    with UnitOfWork(
        "Tool Audit",
        info="Checking for required developer tools (uv, doppler).",
    ) as work:
        tools_to_check = {
            "uv": "uv --version",
            "doppler": "doppler --version",
        }

        missing_tools: list[str] = []
        for tool in tools_to_check:
            if not is_tool_installed(tool):
                work.logger.warning(f"{tool} is NOT installed.")
                missing_tools.append(tool)
            else:
                work.logger.success(f"{tool} is available.")

        if not missing_tools:
            work.logger.success("All core tools are ready to go.")

        return missing_tools


def install_missing_tools(missing_tools: list[str]) -> None:
    """Orchestrate the installation of missing tools based on OS."""
    if not missing_tools:
        return

    os_type = platform.system()

    with UnitOfWork(
        "Tool Installation",
        info=f"Orchestrating installation of: {', '.join(missing_tools)} ({os_type})",
    ) as work:
        if os_type == "Darwin":
            _install_mac(work, missing_tools)
        elif os_type == "Linux":
            _install_linux(work, missing_tools)
        elif os_type == "Windows":
            _install_windows(work, missing_tools)
        else:
            work.logger.error(f"Unsupported OS: {os_type}")


def _install_mac(work: UnitOfWork, tools: list[str]) -> None:
    """Installation logic for macOS."""
    has_brew = is_tool_installed("brew")

    for tool in tools:
        if tool == "uv":
            if has_brew:
                Command(
                    work.logger,
                    cmd="brew install uv",
                    description="Installing uv (Homebrew)",
                )
            else:
                Command(
                    work.logger,
                    cmd="curl -fsSL https://astral.sh/uv/install.sh | sh",
                    description="Installing uv (curl)",
                )
        elif tool == "doppler":
            if has_brew:
                Command(
                    work.logger,
                    cmd="brew install dopplerhq/cli/doppler",
                    description="Installing Doppler CLI (Homebrew)",
                )
            else:
                work.logger.header("Manual Setup Required")
                work.logger.info("Visit: https://docs.doppler.com/docs/install-cli")


def _install_linux(work: UnitOfWork, tools: list[str]) -> None:
    """Installation logic for Linux."""
    for tool in tools:
        if tool == "uv":
            Command(
                work.logger,
                cmd="curl -fsSL https://astral.sh/uv/install.sh | sh",
                description="Installing uv (curl)",
            )
        elif tool == "doppler":
            work.logger.header("Manual Setup Required (Linux)")
            work.logger.info("Visit: https://docs.doppler.com/docs/install-cli")


def _install_windows(work: UnitOfWork, tools: list[str]) -> None:
    """Installation logic for Windows."""
    has_winget = is_tool_installed("winget")

    for tool in tools:
        if tool == "uv":
            if has_winget:
                Command(
                    work.logger,
                    cmd="winget install uv",
                    description="Installing uv (winget)",
                )
            else:
                Command(
                    work.logger,
                    cmd='powershell -ExecutionPolicy ByPass -c "'
                    'irm https://astral.sh/uv/install.ps1 | iex"',
                    description="Installing uv (PowerShell)",
                )
        elif tool == "doppler":
            if has_winget:
                Command(
                    work.logger,
                    cmd="winget install doppler",
                    description="Installing Doppler CLI (winget)",
                )
            else:
                work.logger.warning("Manual setup suggested for Doppler on Windows.")
                work.logger.info("Visit: https://docs.doppler.com/docs/install-cli")
