from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from ..classes import GuiderSequence, ObservationSequence
from ..logger import LOGGER


def _plot_fwhm_sequence(gseq: List[GuiderSequence], ax: Optional[Axes] = None):
    """Helper function to plot FWHM summary for a list of GuiderSequences."""
    ax = ax if ax is not None else plt.gca()
    fwhms = np.array([s.get_fwhm_stats() for s in gseq])
    ax.errorbar(
        range(len(gseq)),
        fwhms[:, 0],
        yerr=fwhms[:, 1],
        fmt="o",
        capsize=5,
        label="FWHM",
    )
    ax.set_xlabel("Observation Index")
    ax.set_ylabel("FWHM (arcsec)")
    ax.set_title("FWHM per Observation")
    ymax = max(1.05 * np.nanmax(fwhms[:, 0]), 3)
    ax.set_ylim(0, ymax)


def _plot_combined_centroids(
    gseq: List[GuiderSequence],
    ax: Optional[Axes] = None,
    dithers: Optional[List[int]] = None,
):
    ax = ax if ax is not None else plt.gca()
    for i, s in enumerate(gseq):
        color = plt.cm.tab10(i % 6)  # type: ignore
        s.plot_centroid_positions(
            "origin",
            ax=ax,
            set_limits=False,
            color=color,
            alpha=0.5,
            dither=dithers[i] if dithers else None,
        )


def plot_guider_sequence_summary(oseq: ObservationSequence):
    gseq = oseq.guider_sequences
    if gseq is None:
        LOGGER.error("Guider sequences not loaded. Call load_guider_sequences() first.")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    t = f"{oseq.targets[0]} ({len(oseq)}), start: {oseq.observations[0].start_time_ut.strftime('%Y-%m-%d %H:%M')}"
    fig.suptitle(t)
    dithers = [obs.dither for obs in oseq.observations]
    _plot_combined_centroids(gseq, ax1, dithers)
    _plot_fwhm_sequence(gseq, ax2)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # type: ignore
