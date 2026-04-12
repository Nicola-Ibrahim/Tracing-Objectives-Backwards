import platform
import shutil
import subprocess
import sys
from pathlib import Path

# --- ANSI Color/Style Constants ---
BOLD = "\033[1m"
DIM = "\033[2m"
BLUE = "\033[94m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

# --- Formatting Constants ---
INDENT_STEP = "  "
INDENT_OUTPUT = "      "

def log_header(text):
    """Prints a bold, high-contrast block header with consistent padding."""
    width = 60
    border = "━" * width
    print(f"\n{BOLD}{BLUE}{border}{RESET}")
    print(f"{BOLD}{BLUE} {text.upper()}{RESET}")
    print(f"{BOLD}{BLUE}{border}{RESET}\n")

def log_step(text):
    """Prints a structured action step with indentation."""
    print(f"{INDENT_STEP}{BOLD}{CYAN}➤ {text}...{RESET}")

def log_success(text):
    """Prints a success message with indentation."""
    print(f"{INDENT_STEP}{BOLD}{GREEN}✔ {text}{RESET}")

def log_error(text):
    """Prints an error message with indentation."""
    print(f"{INDENT_STEP}{BOLD}{RED}✖ ERROR: {text}{RESET}")

def log_warning(text):
    """Prints a warning message with indentation."""
    print(f"{INDENT_STEP}{BOLD}{YELLOW}⚠  {text}{RESET}")

def log_info(text):
    """Prints an informational message with dim styling and indentation."""
    print(f"{INDENT_STEP}{DIM}{CYAN}ℹ {text}{RESET}")


def get_root_dir():
    """Returns the absolute project root directory."""
    return Path(__file__).parent.parent.absolute()


def is_tool_installed(name):
    """Checks if a tool is available in the system PATH."""
    return shutil.which(name) is not None


def run_command(
    command, description=None, cwd=None, interactive=False, exit_on_error=True
):
    """
    Consolidated shell command runner with enhanced formatting and nested indentation.
    """
    if description:
        log_step(description)

    try:
        if interactive:
            # Run without capturing output to allow interactivity
            result = subprocess.run(command, shell=True, cwd=cwd)
        else:
            result = subprocess.run(
                command, shell=True, cwd=cwd, capture_output=True, text=True
            )
            # Only print output if it's not empty
            if result.stdout and result.stdout.strip():
                # Deeply indent output for better visual separation
                indented_output = "\n".join([f"{INDENT_OUTPUT}{DIM}{line}{RESET}" for line in result.stdout.strip().split("\n")])
                print(indented_output)

        if result.returncode == 0:
            if description:
                log_success(f"Successfully finished: {description}")
            return True
        else:
            if description:
                log_error(f"Failed to complete: {description}")
            
            if not interactive and result.stderr:
                log_error("Raw error output:")
                error_lines = "\n".join([f"{INDENT_OUTPUT}{RED}{line}{RESET}" for line in result.stderr.strip().split("\n")])
                print(error_lines)
                
            if exit_on_error:
                sys.exit(result.returncode)
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
            f"Auto-launch unsupported on this OS. Please run: {command} in {cwd} manually."
        )
