
from datetime import timedelta
from typing import List, Optional
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ..calculations.clipping import get_clipping_kept_mask

from ..classes import GuiderSequence, ObservationSequence
from .util import change_time_labels, get_mid_times

def _set_fwhm_ylimits(ax: Axes, fwhm_values: List[float]):
    """Set y-limits for FWHM plot with some padding."""
    if all(np.isnan(fwhm_values)):
        ax.set_ylim(0, 3)
        return
    fwhm_min = min(fwhm_values)
    fwhm_max = max(fwhm_values)
    height = max(fwhm_max - fwhm_min, 0.1)
    y_padding = 0.5 * height
    ymax = max(fwhm_max + y_padding, fwhm_max * 1.3)
    ax.set_ylim(0, top=ymax)

def plot_fwhm_series(
    gseqs: List[GuiderSequence], oseq: ObservationSequence, ax: Optional[Axes] = None
):
    """Helper function to plot FWHM summary for a list of GuiderSequences."""
    ax = ax if ax is not None else plt.gca()
    all_fwhms, all_times = [], []
    for s in gseqs:
        all_fwhms.extend(s.get_fwhms_arcsec(sigmaclip_val=None))
        all_times.extend([f.ut_time for f in s.frames])

    ax.plot(all_times, all_fwhms, "-", color="k", alpha=0.3)
    fwhms = np.array([s.get_fwhm_stats() for s in gseqs])
    mid_times = get_mid_times(oseq.observations)
    xerr = np.array([timedelta(seconds=o.exptime / 2) for o in oseq])
    ax.errorbar(
        mid_times,
        fwhms[:, 0],
        yerr=fwhms[:, 1],
        xerr=xerr,
        fmt="o",
        capsize=5,
        label="Fitted FWHM (sigmaclipped with $2.5\\sigma$)",
    )
    noted_fwms = np.array([o.fwhm_noted for o in oseq.observations])
    mask = ~np.isnan(noted_fwms)
    if np.any(mask):
        ax.scatter(
            np.array(mid_times)[mask],
            noted_fwms[mask],
            marker="x",
            s=20,
            color="red",
            label="FWHM reported in log",
            zorder=3,
        )
    ax.grid(True)
    ax.set_ylabel("FWHM (arcsec)")
    ax.set_title("FWHM")
    change_time_labels(ax, mid_times, oseq.time_range)
    ax.legend(loc="lower left")
    _set_fwhm_ylimits(ax, fwhms[:, 0])

def plot_fwhms_for_single_gseq(
    gseq: GuiderSequence,
    ax: Optional[plt.Axes] = None,
    annotate_mean: bool = True,
    **plot_kwargs,
) -> plt.Axes:
    """Plot FWHM (arcsec) over frame index for a GuiderSequence.

    Parameters
    ----------
    seq : GuiderSequence
        The guider sequence with fitted models.
    ax : plt.Axes, optional
        Matplotlib Axes to plot on. If None, creates a new figure and axes.
    annotate_mean : bool, optional
        Whether to annotate the mean and stddev on the plot. Default is True.
    plot_kwargs : dict
        Additional keyword arguments passed to plt.plot.

    Returns
    -------
    plt.Axes
        The Axes object containing the FWHM plot.
    """
    ax = plt.gca() if ax is None else ax
    plot_kwargs = plot_kwargs.copy()

    mean_fwhm, std_fwhm = gseq.get_fwhm_stats()
    plot_kwargs.setdefault("marker", "x")
    plot_kwargs.setdefault("linestyle", "none")
    plot_kwargs.setdefault("markersize", 10)
    plot_kwargs.setdefault("color", "blue")
    all_fwhms = gseq.get_fwhms_arcsec(sigmaclip_val=None)
    clip_mask = get_clipping_kept_mask(all_fwhms)
    ax.plot(gseq.guider_times[clip_mask], all_fwhms[clip_mask], ls="-", color="gray", lw=0.5)
    ax.plot(gseq.guider_times[clip_mask], all_fwhms[clip_mask], **plot_kwargs)
    plot_kwargs["alpha"] = 0.5
    plot_kwargs["color"] = "red"
    plot_kwargs["linestyle"] = "None"
    plot_kwargs["marker"] = "x"
    plot_kwargs["markersize"] = 8
    ax.plot(gseq.guider_times[~clip_mask], all_fwhms[~clip_mask], **plot_kwargs)
    ts = gseq.observation.timeslot
    if ts is not None:
        change_time_labels(ax, gseq.guider_times, (ts.start_time, ts.end_time))
    _set_fwhm_ylimits(ax, all_fwhms)
    ax.set_xlabel("Time (UT)")
    ax.set_ylabel("FWHM (arcsec)")
    ax.set_title("FWHM")
    ax.grid(True)
    if not annotate_mean:
        return ax
    ax.axhline(mean_fwhm, color="red", linestyle="--")
    ax.axhspan(
        mean_fwhm - std_fwhm,
        mean_fwhm + std_fwhm,
        color="red",
        alpha=0.2,
        label="$1\\sigma$ range",
    )
    ax.text(
        0.95,
        0.02,
        f"Mean FWHM: ${mean_fwhm:.2f}\\pm {std_fwhm:.2f}$ arcsec",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="red",
    )

    return ax
