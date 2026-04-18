from .engine.ui import Logger
from .tasks.env_secrets import setup_identity
from .tasks.setup import check_core_tools, install_missing_tools


def main() -> bool:
    """Entry point for the environment bootstrap process."""
    logger = Logger()
    logger.header("Environment Bootstrap")
    logger.info("Ensuring all core developer tools are installed and configured.")

    # 1. Audit core tools (uv, doppler)
    missing = check_core_tools()

    # 2. Install if needed
    install_missing_tools(missing)

    # 3. Setup identity
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
