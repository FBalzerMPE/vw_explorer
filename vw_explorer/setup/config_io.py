from pathlib import Path
from typing import Dict, Optional

import yaml

from ..logger import LOGGER

_VWE_DIR = Path.home() / ".vw_explorer"
_USER_CONFIG_PATH = _VWE_DIR / "config.yml"
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "assets" / "default_config.yml"


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


def _ask_user_to_create_path(path: Path, key: str) -> bool:
    """Asks the user whether to create a missing path."""
    if ask_user_confirmation(
        f"The path for '{key}' ('{path}')\ndoes not exist. Create it?"
    ):
        path.mkdir(parents=True, exist_ok=True)
        return True
    return False


def create_missing_paths(config: dict):
    """Creates any missing paths specified in the config."""
    for key, path in config["paths"].items():
        p = Path(path)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=False)
            LOGGER.info(f"Created missing path for '{key}' at {path}")


def validate_config(config: dict):
    for key, path in config["paths"].items():
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Path for '{key}' does not exist: {path}\nEither create it, or change it in the config file (vw_explorer/config.yml)."
            )


def sanitize_path(path: str) -> Path:
    """Converts a string path to a Path object and resolves it."""
    path = path.replace(r"{VWE}", str(_VWE_DIR))
    return Path(path).expanduser().resolve()


def load_config(validate: bool = True) -> Dict[str, Dict]:
    """
    Loads the configuration file.

    Parameters
    ----------
    config_path : Optional[Path]
        Path to the configuration file. If None, the default config file is used.
    validate : bool
        Whether to validate the paths in the configuration file.

    Returns
    -------
    Dict[str, Dict]
        The loaded configuration.
    """
    if not _USER_CONFIG_PATH.exists():
        if not ask_user_confirmation(
            f"No user configuration file found at {_USER_CONFIG_PATH}.\nWould you like to create a default configuration file there?"
        ):
            raise FileNotFoundError(
                f"Configuration file not found: {_USER_CONFIG_PATH}"
            )
        generate_default_config()

    config_file = Path(_USER_CONFIG_PATH)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {_USER_CONFIG_PATH}")

    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    for key, path in cfg["paths"].items():
        cfg["paths"][key] = str(sanitize_path(path))

    if validate:
        try:
            validate_config(cfg)
        except FileNotFoundError as e:
            LOGGER.error(e)
            q = "Would you like to create the necessary directories specified in the config file? Otherwise, first edit the config file to set existing paths."
            if ask_user_confirmation(q):
                cfg = load_config(validate=False)
                create_missing_paths(cfg)
                validate_config(cfg)

    return cfg


def generate_default_config():
    """
    Generates a default configuration file in the user's home directory.
    """
    if not _DEFAULT_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Default configuration file not found: {_DEFAULT_CONFIG_PATH}"
        )

    config_path = _USER_CONFIG_PATH

    # Ensure the parent directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)
    # Copy the default configuration to the new location
    config_path.write_text(_DEFAULT_CONFIG_PATH.read_text())
    LOGGER.info(f"Default configuration file created at: {config_path}")
