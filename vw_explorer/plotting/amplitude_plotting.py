from typing import List, Optional
from datetime import timedelta
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ..classes import GuiderSequence, ObservationSequence
from .util import change_time_labels, get_mid_times

def plot_amplitude_series(
    gseq: List[GuiderSequence], oseq: ObservationSequence, ax: Optional[Axes] = None
):
    """Helper function to plot Amplitude summary for a list of GuiderSequences."""
    ax = ax if ax is not None else plt.gca()
    all_amplitudes, all_times = [], []
    for s in gseq:
        all_amplitudes.extend(s.get_amplitudes(sigmaclip_val=None))
        all_times.extend([f.ut_time for f in s.frames])

    ax.plot(all_times, all_amplitudes, "-", color="k", alpha=0.3)

    amplitudes = np.array([s.get_amplitude_stats() for s in gseq])
    mid_times = get_mid_times(oseq.observations)
    xerr = np.array([timedelta(seconds=o.exptime/2) for o in oseq])

    ax.errorbar(
        mid_times,
        amplitudes[:, 0],
        xerr=xerr,
        yerr=amplitudes[:, 1],
        fmt="o",
        capsize=5,
        label="Mean Amplitude (no sigmaclip)",
        color="green",
    )
    
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title("Amplitude of guide star fit")
    change_time_labels(ax, mid_times, oseq.time_range)
    ax.legend(loc="lower left")
    ymax = 1.1 * np.nanpercentile(all_amplitudes, 98)
    ax.set_ylim(0, ymax)


def _set_amplitude_ylimits(ax: Axes, amplitude_values: List[float]):
    """Set y-limits for Amplitude plot with some padding."""
    if all(np.isnan(amplitude_values)):
        ax.set_ylim(0, 1)
        return
    amp_min = min(amplitude_values)
    amp_max = max(amplitude_values)
    height = max(amp_max - amp_min, 0.1)
    y_padding = 0.2 * height
    ymin = max(0, amp_min - y_padding)
    ymax = amp_max + y_padding
    ax.set_ylim(ymin, ymax)


def plot_amplitudes_for_single_gseq(
    gseq: GuiderSequence,
    annotate_mean: bool = True,
    ax: Optional[plt.Axes] = None,
    **plot_kwargs,
) -> plt.Axes:
    """Plots the amplitude as a function of time.

    Parameters
    ----------
    gseq : GuiderSequence
        The guider sequence with fitted models.
    annotate_mean : bool, optional
        Whether to annotate the mean and stddev on the plot. Default is True.
    ax : plt.Axes, optional
        Matplotlib Axes to plot on. If None, creates a new figure and axes.
    plot_kwargs : dict
        Additional keyword arguments passed to plt.plot.

    Returns
    -------
    plt.Axes
        The Axes object containing the amplitude plot.
    """
    ax = plt.gca() if ax is None else ax

    amplitudes = gseq.get_amplitudes(sigmaclip_val=None)
    plot_kwargs.setdefault("markersize", 10)
    plot_kwargs.setdefault("color", "gray")
    plot_kwargs.setdefault("markerfacecolor", "k")
    plot_kwargs.setdefault("marker", "o")
    plot_kwargs.setdefault("linewidth", 0.5)

    ax.plot(gseq.guider_times, amplitudes, **plot_kwargs)
    _set_amplitude_ylimits(ax, amplitudes)
    ax.set_xlabel("Time (UT)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Amplitude")
    ax.grid(True)
    ts = gseq.observation.timeslot
    if ts is not None:
        change_time_labels(ax, gseq.guider_times, (ts.start_time, ts.end_time))

    if not annotate_mean:
        return ax
    mean_amp = np.nanmean(amplitudes)
    std_amp = np.nanstd(amplitudes)
    ax.axhline(mean_amp, color="red", linestyle="--")
    ax.axhspan(
        mean_amp - std_amp,
        mean_amp + std_amp,
        color="red",
        alpha=0.2,
        label="$1\\sigma$ range",
    )
    ax.text(
        0.95,
        0.02,
        f"Mean Amplitude: ${mean_amp:.2f}\\pm {std_amp:.2f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="red",
    )

    return ax
