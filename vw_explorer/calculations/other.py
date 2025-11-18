from collections import Counter
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ..classes import Observation

from ..constants import CALIB_NAMES


def get_target_counts(
    observations: List["Observation"], remove_calib: bool = True
) -> Counter:
    """Returns a Counter of target occurrences in the observations.
    Filtering out calibration targets if remove_calib is True.

    Parameters
    ----------
    observations : List["Observation"]
        List of Observation objects to count targets from.
    remove_calib : bool, optional
        Whether to exclude calibration targets from the count, by default True.

    Returns
    -------
    Counter
        A Counter object with target names as keys and their counts as values.
    """
    if remove_calib:
        observations = [obs for obs in observations if obs.target not in CALIB_NAMES]
    return Counter(obs.target for obs in observations)
