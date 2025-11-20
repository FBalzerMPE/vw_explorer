from pathlib import Path
from typing import List, Tuple

import pandas as pd

from ...classes import GuiderSequence, Observation
from ...logger import LOGGER


def load_guider_sequences(observations: List[Observation]) -> List[GuiderSequence]:
    """
    Processes a list of Observations to compute derived information.

    Parameters
    ----------
    observations : List[Observation]
        List of Observation objects.

    Returns
    -------
    List[GuiderSequence]
        A list of GuiderSequence objects.
    """
    guider_sequences = []
    for obs in observations:
        if not obs.is_sky_obs:
            continue
        try:
            g = GuiderSequence(obs)
            if len(g) <= 1:
                LOGGER.warning(
                    f"Skipping observation {obs.filename} with only {len(g)} guider frames."
                )
                continue
            guider_sequences.append(g)
        except Exception as e:
            LOGGER.warning(f"Error processing observation {obs.filename}: {e}")
    return guider_sequences
