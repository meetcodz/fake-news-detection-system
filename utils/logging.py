import logging
from typing import Any

def setup_logging(config: dict[str, Any] | None = None) -> None:
    """Configure the root logger from an optional config mapping."""
    log_config = (config or {}).get("logging", {})
    level_name = str(log_config.get("level", "INFO")).upper()
    log_format = log_config.get(
        "format",
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    level = getattr(logging, level_name, logging.INFO)
    root_logger = logging.getLogger()

    if root_logger.handlers:
        root_logger.setLevel(level)
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name)
