from pathlib import Path
from typing import Optional

from .logger import LOGGER


def sanitize_obs_fpath(f_in: Optional[str]) -> Path:
    if f_in is not None:
        if (p := Path(f_in)).exists():
            return p
        fpath = Path(f"vw{f_in:0>6}.fits")
        if fpath.exists():
            return fpath
        raise FileNotFoundError(f"File {f_in} not found.")
    fits_files = sorted(Path(".").glob("vw*.fits"), key=lambda x: x.stat().st_mtime)
    if not fits_files:
        LOGGER.error(
            "No FITS files found matching 'vw*.fits'. Please provide an input file."
        )
        raise FileNotFoundError("No FITS files following 'vw*.fits' format found.")
    return fits_files[-1]


def retrieve_file_paths(f_in: Optional[str]) -> list[Path]:
    if f_in is None:
        fits_files = sorted(Path(".").glob("vw*.fits"), key=lambda x: x.stat().st_mtime)
        if not fits_files:
            LOGGER.error(
                "No FITS files found matching 'vw*.fits'. Please provide an input file."
            )
            raise FileNotFoundError("No FITS files following 'vw*.fits' format found.")
        return fits_files
    p = Path(f_in)
    if p.exists():
        return [p]
    else:
        raise FileNotFoundError(f"File {f_in} not found.")
