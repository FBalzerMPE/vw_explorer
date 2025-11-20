from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

from ...classes import GuiderSequence, Observation
from ...logger import LOGGER


def generate_observation_plots(
    observations: List[Observation],
    guider_sequences: List[GuiderSequence],
    output_dir: Path,
):
    """
    Generates and saves plots for each observation.

    Parameters
    ----------
    observations : List[Observation]
        List of Observation objects.
    output_dir : Path
        Directory to save the plots.
    """
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for obs, gseq in zip(observations, guider_sequences):
        if not obs.is_sky_obs:
            continue
        try:
            fig, axes = plt.subplots(1, 3, figsize=(10, 5))
            gseq.plot_centroid_positions("fiducial", ax=axes[0])
            gseq.plot_fwhm_timeseries(ax=axes[1])
            fig.suptitle(f"Observation: {obs.long_name}", fontsize=16)
            plot_path = plot_dir / f"{obs.filename}_summary.png"
            fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            LOGGER.info(f"Saved plot for {obs.filename} to {plot_path}")
        except Exception as e:
            LOGGER.warning(f"Error generating plot for {obs.filename}: {e}")
