import shutil
import subprocess
import sys
from pathlib import Path

from .ui import logger


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
            if not logger.confirm(f"Run step: {description}?"):
                logger.info_step("Skipped by user.")
                return False

        logger.step(description)

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
                    logger.output(line)

            process.wait()
            returncode = process.returncode
        else:
            # Batch mode
            result = subprocess.run(
                command, shell=True, cwd=cwd, capture_output=True, text=True
            )
            if result.stdout and result.stdout.strip():
                logger.output(result.stdout)

            if result.returncode != 0 and result.stderr:
                logger.error("Raw error output:")
                logger.error_output(result.stderr)

            returncode = result.returncode

        if returncode == 0:
            if description:
                logger.success(f"Successfully finished: {description}")
            return True
        else:
            if description:
                logger.error(f"Failed to complete: {description}")

            if exit_on_error:
                sys.exit(returncode)
            return False

    except Exception as e:
        if description:
            logger.error(f"Critical exception during: {description}")
        logger.info_step(f"Exception: {str(e)}")
        if exit_on_error:
            sys.exit(1)
        return False
