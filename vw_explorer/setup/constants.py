from pathlib import Path
VWE_CONFIG_DIR = Path.home() / ".vw_explorer"
USER_CONFIG_PATH = VWE_CONFIG_DIR / "config.yml"
VWE_DIR = Path(__file__).parent.parent
DEFAULT_CONFIG_PATH = VWE_DIR / "assets" / "default_config.yml"