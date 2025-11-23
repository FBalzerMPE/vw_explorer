""" "Calculations involving clipping of outlier data points."""

from typing import Optional

import numpy as np
from astropy.stats import sigma_clip


def get_clipping_kept_mask_by_distance(
    centroids: np.ndarray, sigmaclip_val: Optional[float] = 2.5, **kwargs
) -> np.ndarray:
    """Returns a boolean mask indicating which (x, y) centroids are kept after sigma-clipping.
    If sigmaclip_val is None, all centroids are kept (mask of all True).
    """
    if sigmaclip_val is None:
        return np.ones(centroids.shape[0], dtype=bool)
    if len(centroids) == 0:
        return np.array([], dtype=bool)
    if centroids.shape[0] == 1:
        return np.array([True], dtype=bool)

    med = np.median(centroids, axis=0)
    d = np.hypot(centroids[:, 0] - med[0], centroids[:, 1] - med[1])
    clipped = sigma_clip(d, sigma=sigmaclip_val, **kwargs)  # type: ignore
    good_mask = ~clipped.mask  # type: ignore
    return good_mask


def get_clipping_kept_mask(
    values: np.ndarray, sigmaclip_val: Optional[float] = 2.5, **kwargs
) -> np.ndarray:
    """Returns a boolean mask indicating which values are kept after sigma-clipping.

    If sigmaclip_val is None, all values are kept (mask of all True).
    """
    if sigmaclip_val is None:
        return np.ones(values.shape, dtype=bool)

    clipped = sigma_clip(values, sigma=sigmaclip_val, **kwargs)  # type: ignore
    good_mask = ~clipped.mask  # type: ignore
    return good_mask
