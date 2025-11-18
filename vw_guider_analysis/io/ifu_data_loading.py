from pathlib import Path
from typing import Tuple

import numpy as np
from astropy.io import fits

from ..logger import LOGGER


def _get_fiberpos() -> np.ndarray:
    fpath = Path(__file__).parent.parent / "IFUcen.txt"
    if not fpath.exists():
        raise FileNotFoundError(f"Fiber position file not found at {fpath}")
    fiberpos = np.loadtxt(fpath, comments="#")
    # Flip to match finderchart/guider
    fiberpos[:, 2] = fiberpos[:, 2] * -1.0
    return fiberpos


def load_ifu_data(fits_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the FITS data and extracts fluxes for each fiber.
    """
    ycut = 2048  # y position to cut at
    xw, yw = 6.0, 300.0  # extraction box width
    kappa = 2.0  # Outlier rejection threshold

    LOGGER.info(f"Reading {fits_path}")
    fiberpos = _get_fiberpos()

    img: np.ndarray = fits.getdata(fits_path, ignore_missing_end=True)  # type: ignore
    if img.shape[0] < 4096:
        # assume y binned
        ycut /= 2
        yw /= 2
    if img.shape[1] < 2048:
        # assume x binned
        LOGGER.error("Unfortunately quicklook only works on non x binned images!")
        raise ValueError("Input image is x binned, which is not supported.")

    imA = img[0:, 0:1025]
    imA = imA - np.median(img[0:, 1030:1060])
    imB = img[0:, 1124:]
    imB = imB - np.median(img[0:, 1090:1120])

    img = np.concatenate((imA, imB), axis=1)

    flux = np.zeros(fiberpos.shape[0])

    # Collapse along fibers
    for i in range(fiberpos.shape[0]):
        xmin, xmax = int(fiberpos[i, 3] - xw / 2.0), int(fiberpos[i, 3] + xw / 2.0)
        ymin, ymax = int(ycut - yw / 2.0), int(ycut + yw / 2)
        f = (img[ymin:ymax, xmin:xmax]).flatten()

        mdflux = np.median(f[~np.isnan(f)])
        # Remove outliers
        f[abs(f - mdflux) > kappa * np.std(f)] = np.nan
        mnflux = np.mean(f[~np.isnan(f)])
        flux[i] = mnflux
        # img[ymin:ymax, xmin:xmax] = 0
    return fiberpos, flux
