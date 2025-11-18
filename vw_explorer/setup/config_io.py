from pathlib import Path
from typing import Optional

import yaml


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


def validate_config(config: dict):
    for key, path in config["paths"].items():
        if not Path(path).exists():
            if not _ask_user_to_create_path(Path(path), key):
                raise FileNotFoundError(
                    f"Path for '{key}' does not exist: {path}\nEither create it, or change it in the config file (vw_explorer/config.yml)."
                )


def sanitize_path(path: str) -> Path:
    """Converts a string path to a Path object and resolves it."""
    path = path.replace(r"{VWE}", str(Path(__file__).parent.parent.parent))
    return Path(path).expanduser().resolve()


def load_config(
    config_path: Optional[Path] = None, validate: bool = True
) -> dict[str, dict]:
    """
    Loads the configuration file.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config.yml"
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)
    for key, path in cfg["paths"].items():
        cfg["paths"][key] = str(sanitize_path(path))
    if validate:
        validate_config(cfg)
    return cfg
