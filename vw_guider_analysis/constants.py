from pathlib import Path

GUIDER_PIXSCALE = 0.533
"""The pixel scale of the VW guider camera in arcseconds per pixel."""
DATA_DIR = Path(__file__).parent.parent / "data"
"""The path to the data directory."""
GUIDER_DIR = DATA_DIR / "guider_frames"
"""The path to the guider frames directory."""

CALIB_NAMES = [
    "biases",
    "autofocus",
    "domeflats",
    "arcs",
    "test",
    "skyflats",
    "twilight",
    "twilights",
]
"""List of standard calibration observation names."""
