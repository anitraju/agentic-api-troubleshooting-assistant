"""Application entry point for the initial project skeleton."""

import logging

from app.config import configure_logging, get_settings

logger = logging.getLogger(__name__)


def main() -> None:
    """Start the application and verify that configuration loads correctly."""
    settings = get_settings()
    configure_logging(settings.log_level)

    logger.info("Starting %s", settings.app_name)
    logger.info("Environment: %s", settings.app_env)
    logger.info("Project initialization completed successfully.")


if __name__ == "__main__":
    main()
