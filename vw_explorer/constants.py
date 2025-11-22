from pathlib import Path

from .setup import CONFIG

GUIDER_PIXSCALE = 0.533
"""The pixel scale of the VW guider camera in arcseconds per pixel."""
ROOT_DIR = Path(__file__).parent
ASSET_PATH = ROOT_DIR / "assets"
DATA_PATH = Path(CONFIG["paths"]["data_dir"])
"""The path to the data directory."""
GUIDER_PATH = Path(CONFIG["paths"]["guider_dir"])
"""The path to the guider frames directory."""
OBS_PATH = Path(CONFIG["paths"]["observation_dir"])
"""The path to the observation FITS files directory."""
OUTPUT_PATH = Path(CONFIG["paths"]["output_dir"])
"""The path to the output directory."""

CALIB_NAMES = [
    "unknown",
    "biases",
    "bias",
    "autofocus",
    "domeflats",
    "domeflat",
    "flats",
    "arcs",
    "skyflats",
    "twilight",
    "twilights",
    "test",
    "tests",
]
"""List of standard calibration observation names."""
