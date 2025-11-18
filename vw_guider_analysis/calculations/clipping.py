""" "Calculations involving clipping of outlier data points."""

from typing import Optional

import numpy as np
from astropy.stats import sigma_clip


def get_mean_centroid(
    centroids: np.ndarray, sigmaclip_val: Optional[float] = 3
) -> np.ndarray:
    """Returns the mean (x, y) centroid from a set of centroids in the shape (N, 2).

    If sigmaclip_val is given, perform sigma-clipping on the radial distance
    from the median centroid to reject outlier points as whole (x,y) tuples.
    """
    if centroids.size == 0:
        return np.array([np.nan, np.nan])

    if sigmaclip_val is None:
        return np.mean(centroids, axis=0)

    # robust center to measure distances from
    med = np.median(centroids, axis=0)

    # radial distances from the median point
    d = np.hypot(centroids[:, 0] - med[0], centroids[:, 1] - med[1])

    # sigma-clip distances and keep rows that are not masked
    clipped = sigma_clip(d, sigma=sigmaclip_val)  # type: ignore
    good_mask = ~clipped.mask  # type: ignore
    if np.count_nonzero(good_mask) == 0:
        return med

    return np.mean(centroids[good_mask], axis=0)


def get_mean_value(values: np.ndarray, sigmaclip_val: Optional[float] = 3) -> float:
    """Returns the mean value from a set of values in the shape (N,).

    If sigmaclip_val is given, perform sigma-clipping to reject outlier points.
    """
    if values.size == 0:
        return float("nan")

    if sigmaclip_val is None:
        return float(np.mean(values))

    clipped = sigma_clip(values, sigma=sigmaclip_val)  # type: ignore
    good_mask = ~clipped.mask  # type: ignore
    if np.count_nonzero(good_mask) == 0:
        return float(np.median(values))

    return float(np.mean(values[good_mask]))
