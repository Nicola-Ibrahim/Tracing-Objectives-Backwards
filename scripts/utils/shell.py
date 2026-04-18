import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .ui import (
    DIM,
    INDENT_OUTPUT,
    RED,
    RESET,
    confirm_action,
    log_error,
    log_step,
    log_success,
    log_warning,
)


def get_root_dir():
    """Returns the absolute project root directory."""
    # Assuming the scripts directory is at the root or one level deep
    # Current structure: project/scripts/lib/shell.py
    return Path(__file__).parent.parent.parent.absolute()


def is_tool_installed(name):
    """Checks if a tool is available in the system PATH."""
    return shutil.which(name) is not None


def run_command(
    command: str,
    description: str | None = None,
    cwd: str | Path | None = None,
    interactive: bool = False,
    stream: bool = False,
    confirm: bool = False,
    skip_confirm: bool = False,
    exit_on_error: bool = False,
) -> bool:
    """
    Consolidated shell command runner with enhanced formatting,
    optional real-time streaming, and granular step-by-step
    confirmation prompts.
    """
    if description:
        # If confirmation is requested, check skip_confirm or ask user
        if confirm and not skip_confirm:
            if not confirm_action(f"Run step: {description}?"):
                print(f"{INDENT_OUTPUT}{DIM}Skipped by user.{RESET}")
                return False

        log_step(description)

    try:
        if interactive:
            # Run without capturing output to allow interactivity
            result = subprocess.run(command, shell=True, cwd=cwd)
            returncode = result.returncode
        elif stream:
            # Stream output in real-time
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            for line in process.stdout:
                if line.strip():
                    print(f"{INDENT_OUTPUT}{DIM}{line.strip()}{RESET}")

            process.wait()
            returncode = process.returncode
        else:
            # Batch mode
            result = subprocess.run(
                command, shell=True, cwd=cwd, capture_output=True, text=True
            )
            if result.stdout and result.stdout.strip():
                indented_output = "\n".join(
                    [
                        f"{INDENT_OUTPUT}{DIM}{line}{RESET}"
                        for line in result.stdout.strip().split("\n")
                    ]
                )
                print(indented_output)

            if result.returncode != 0 and result.stderr:
                log_error("Raw error output:")
                error_lines = "\n".join(
                    [
                        f"{INDENT_OUTPUT}{RED}{line}{RESET}"
                        for line in result.stderr.strip().split("\n")
                    ]
                )
                print(error_lines)

            returncode = result.returncode

        if returncode == 0:
            if description:
                log_success(f"Successfully finished: {description}")
            return True
        else:
            if description:
                log_error(f"Failed to complete: {description}")

            if exit_on_error:
                sys.exit(returncode)
            return False

    except Exception as e:
        if description:
            log_error(f"Critical exception during: {description}")
        print(f"{INDENT_OUTPUT}Exception: {str(e)}")
        if exit_on_error:
            sys.exit(1)
        return False


def launch_terminal_tab(command, title, cwd):
    """Launch a new terminal tab on macOS and run a command."""
    if platform.system() == "Darwin":
        applescript = f"""
        tell application "Terminal"
            activate
            do script "cd {cwd} && {command}"
        end tell
        """
        subprocess.run(["osascript", "-e", applescript])
        log_success(f"Service '{title}' launched in a new terminal tab.")
    else:
        log_warning(
            f"Auto-launch unsupported on this OS. Please run: {command} "
            f"in {cwd} manually."
        )
