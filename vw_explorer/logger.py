import logging
import sys
from pathlib import Path

from colorama import Fore, Style, init
from typing import Optional, Union

# Initialize colorama for Windows compatibility
init(autoreset=True)

# ANSI escape codes for colors
_LOG_COLORS = {
    "DEBUG": Fore.BLUE,  # Blue
    "INFO": Fore.GREEN,  # Green
    "WARNING": Fore.YELLOW,  # Yellow
    "ERROR": Fore.RED,  # Red
    "CRITICAL": Fore.MAGENTA,  # Magenta
    "RESET": Style.RESET_ALL,  # Reset to default
}
_BOLD = Style.BRIGHT


class ColorFormatter(logging.Formatter):
    """Custom formatter to add colors to log levels."""

    def format(self, record):
        log_color = _LOG_COLORS.get(record.levelname, _LOG_COLORS["RESET"])
        record.levelname = f"{_BOLD}{log_color}{record.levelname}{_LOG_COLORS['RESET']}"
        return super().format(record)


def setup_logger(
    name,
    log_file: Optional[Union[str, Path]] = None,
    level=logging.INFO,
    fmt_colors: bool = True,
):
    """Return a logger with the specified name and level.

    If a log file is specified, the logger will write to that file. Otherwise, it will write to
    stdout.

    Parameters
    ----------
    name : str
        The name of the logger.
    log_file : str, optional
        The path to the log file. If not specified, the logger will write to stdout.
    level : int, optional
        The logging level.

    Returns
    -------
    logging.Logger
        The logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt_str = "[%(asctime)s %(name)s: %(levelname)s] - %(message)s"
    date_fmt = "%H:%M:%S"

    if fmt_colors and log_file is None and sys.stdout.isatty():
        # isatty checks for ANSI support to display colors
        formatter = ColorFormatter(fmt_str, datefmt=date_fmt)
    else:
        formatter = logging.Formatter(fmt_str, datefmt=date_fmt)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file)
    else:
        handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


LOGGER = setup_logger("vw-explorer")
LOGGER.setLevel(logging.INFO)
