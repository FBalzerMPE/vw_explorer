from pathlib import Path
from typing import Dict

import yaml

from ..logger import LOGGER
from .constants import DEFAULT_CONFIG_PATH, USER_CONFIG_PATH, VWE_DIR


def ask_user_confirmation(message: str) -> bool:
    """Asks the user for a yes/no confirmation."""
    while True:
        response = input(f"{message} (y/n): ").strip().lower()
        if response in {"y", "yes"}:
            return True
        elif response in {"n", "no"}:
            return False
        else:
            print(f"{message}: Please enter 'y' or 'n'.")


def create_missing_paths(config: dict):
    """Creates any missing paths specified in the config."""
    for key, path in config["paths"].items():
        p = Path(path)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=False)
            LOGGER.info(f"Created missing path for '{key}' at {path}")



def sanitize_path(path: str) -> Path:
    """Converts a string path to a Path object and resolves it."""
    path = path.replace(r"{VWE}", str(VWE_DIR.parent.absolute()))
    return Path(path).expanduser().resolve()




def generate_default_config():
    """
    Generates a default configuration file in the user's home directory.
    """
    if not DEFAULT_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Default configuration file not found: {DEFAULT_CONFIG_PATH}"
        )

    config_path = USER_CONFIG_PATH

    # Ensure the parent directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)
    # Copy the default configuration to the new location
    config_path.write_text(DEFAULT_CONFIG_PATH.read_text())
    LOGGER.info(f"Default configuration file created at: {config_path}")
