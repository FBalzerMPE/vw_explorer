from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from vw_explorer.classes import GuiderSequence, Observation
from vw_explorer.classes.observation_sequence import ObservationSequence
from vw_explorer.constants import OUTPUT_PATH
from vw_explorer.io import parse_or_load_observations
from vw_explorer.logger import LOGGER


def process_dither_chunks(
    observations: List[Observation],
) -> Dict[str, List[ObservationSequence]]:
    """
    Groups observations into dither chunks and processes them.

    Parameters
    ----------
    observations : List[Observation]
        List of Observation objects.

    Returns
    -------
    Dict[str, List[ObservationSequence]]
        A dictionary mapping target names to lists of ObservationSequence objects grouped by dither chunks.
    """
    sequences = {}
    for target in set(obs.target for obs in observations):
        try:
            seq = ObservationSequence.from_target(target, observations)
            chunks = seq.get_all_dither_chunks()
            sequences[target] = chunks
        except Exception as e:
            LOGGER.warning(f"Error processing dither chunks for target {target}: {e}")
    return sequences


def process_observations(
    obs_list: List["Observation"], outpath: Optional[Path] = None
) -> pd.DataFrame:
    """Processes a list of Observations and saves to CSV."""

    if outpath is None:
        outpath = OUTPUT_PATH / "observations_processed.csv"

    workdir = OUTPUT_PATH / "obs_processing_plots"
    workdir.mkdir(parents=True, exist_ok=True)
    obs_df = Observation.to_dataframe(obs_list)
    seqs = []
    for obs in obs_list:
        if not obs.is_sky_obs:
            continue
        g = GuiderSequence(obs)
        if len(g) <= 1:
            LOGGER.warning(
                f"Skipping observation {obs.filename} with only {len(g)} guider frames."
            )
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        g.plot_centroid_positions("fiducial", ax=axes[0])
        g.plot_fwhm_timeseries(ax=axes[1])
        fig.suptitle(f"Observation: {obs.long_name}", fontsize=16)
        opath = workdir / f"{obs.filename}_summary.png"
        fig.savefig(str(opath), dpi=150, bbox_inches="tight")
        plt.close(fig)
        seqs.append(g)
    seqs_df = GuiderSequence.get_combined_stats_df(seqs)
    final_df = obs_df.merge(seqs_df, on="uid", how="left")
    final_df.to_csv(outpath, index=False)
    return final_df
