import logging
import logging.handlers
import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LoggingSettings:
    level: str = "INFO"
    log_dir: str = "logs"
    app_log_filename: str = "app.log"
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


def _ensure_log_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def setup_logging(settings: Optional[LoggingSettings] = None) -> None:
    """Configure application-wide logging.

    - Console handler (human-friendly)
    - Rotating file handler (persistent)

    Idempotent: calling multiple times won't duplicate handlers.
    """

    settings = settings or LoggingSettings(
        level=os.getenv("LOG_LEVEL", "INFO"),
        log_dir=os.getenv("LOG_DIR", "logs"),
    )

    _ensure_log_dir(settings.log_dir)

    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.level.upper(), logging.INFO))

    # Prevent duplicate handlers if setup_logging() is called multiple times.
    if getattr(root, "_configured_by_app", False):
        return

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(root.level)
    console.setFormatter(fmt)

    file_path = os.path.join(settings.log_dir, settings.app_log_filename)
    file_handler = logging.handlers.RotatingFileHandler(
        file_path,
        maxBytes=settings.max_bytes,
        backupCount=settings.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(root.level)
    file_handler.setFormatter(fmt)

    root.addHandler(console)
    root.addHandler(file_handler)

    root._configured_by_app = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
