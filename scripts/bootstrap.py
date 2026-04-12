import platform
import subprocess

from utils import log_header, log_step, log_success, log_info, log_error, log_warning, is_tool_installed, run_command


def main():
    log_header("Environment Bootstrap")
    log_info("Ensuring all core developer tools are installed and configured.")

    os_type = platform.system()
    tools_to_check = {
        "uv": "uv --version",
        "pnpm": "pnpm --version",
        "doppler": "doppler --version",
    }

    log_header("Checking Installed Tools")
    missing_tools = []
    for tool in tools_to_check:
        if not is_tool_installed(tool):
            log_warning(f"{tool} is NOT installed.")
            missing_tools.append(tool)
        else:
            log_success(f"{tool} is available.")

    if not missing_tools:
        log_success("All core tools are ready to go.")
    else:
        log_header(f"Installing missing tools: {', '.join(missing_tools)}")

        # Linux / macOS (Unix-like)
        if os_type in ["Linux", "Darwin"]:
            has_brew = is_tool_installed("brew")

            for tool in missing_tools:
                if tool == "pnpm":
                    if os_type == "Darwin" and has_brew:
                        run_command("brew install pnpm", "Installing pnpm package manager (Homebrew)")
                    else:
                        run_command(
                            "curl -fsSL https://get.pnpm.io/install.sh | sh",
                            "Installing pnpm package manager (curl)",
                        )

                elif tool == "uv":
                    if os_type == "Darwin" and has_brew:
                        run_command("brew install uv", "Installing uv Python tool (Homebrew)")
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
                        log_error("Manual installation required for Doppler on this system.")
                        log_info("Visit: https://docs.doppler.com/docs/install-cli")

        # Windows
        elif os_type == "Windows":
            has_winget = is_tool_installed("winget")

            for tool in missing_tools:
                if tool == "pnpm":
                    if has_winget:
                        run_command("winget install pnpm", "Installing pnpm package manager (winget)")
                    else:
                        run_command(
                            "curl -fsSL https://get.pnpm.io/install.sh | sh",
                            "Installing pnpm package manager (curl)",
                        )

                elif tool == "uv":
                    if has_winget:
                        run_command("winget install uv", "Installing uv Python tool (winget)")
                    else:
                        run_command(
                            'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"',
                            "Installing uv Python tool (PowerShell)",
                        )

                elif tool == "doppler":
                    if has_winget:
                        run_command(
                            "winget install doppler", "Installing Doppler secrets (winget)"
                        )
                    else:
                        log_warning("Manual installation suggested for Doppler on Windows.")
                        log_info("Visit: https://docs.doppler.com/docs/install-cli")

    # Post-check: ensure symbols are in PATH (often requires shell restart, but we can try)
    log_header("Identity & Secrets")
    if is_tool_installed("doppler"):
        # Check if already logged in
        auth_check = subprocess.run(["doppler", "me"], capture_output=True, text=True)
        if auth_check.returncode == 0:
            log_success("Already authenticated with Doppler CLI.")
        else:
            log_info("Launching Doppler interactive login flow.")
            run_command("doppler login", "Logging into Doppler CLI", interactive=True)
    else:
        log_error("Skip Doppler login: Doppler CLI not found in PATH.")

    log_header("Environment Setup Complete")
    log_success("Bootstrap finished successfully!")
    log_info("NEXT STEP: Run 'python3 scripts/setup_dev.py' to launch your daily development workspace.")


if __name__ == "__main__":
    main()
