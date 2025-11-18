from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from ..constants import DATA_DIR
from ..logger import LOGGER

if TYPE_CHECKING:
    from ..classes import Observation


def process_observations(
    obs_list: List["Observation"], outpath: Optional[Path] = None
) -> pd.DataFrame:
    """Processes a list of Observations and saves to CSV."""
    from ..classes import GuiderSequence, Observation

    if outpath is None:
        outpath = DATA_DIR / "observations_processed.csv"

    workdir = outpath.parent / "obs_processing_plots"
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
        g.plot_positions("fiducial", ax=axes[0])
        g.plot_fwhm_timeseries(ax=axes[1])
        fig.suptitle(f"Observation: {obs.long_name}", fontsize=16)
        opath = workdir / f"{obs.filename}_summary.png"
        fig.savefig(str(opath), dpi=150, bbox_inches="tight")
        plt.close(fig)
        seqs.append(g)
    seqs_df = GuiderSequence.to_dataframe(seqs)
    final_df = obs_df.merge(seqs_df, on="uid", how="left")
    final_df.to_csv(outpath, index=False)
    return final_df
