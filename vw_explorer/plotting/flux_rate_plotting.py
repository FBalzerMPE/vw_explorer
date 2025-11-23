from datetime import timedelta
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from ..classes import GuiderSequence, ObservationSequence
from .util import change_time_labels, get_mid_times


def plot_flux_rate_series(
    gseq: List[GuiderSequence], oseq: ObservationSequence, ax: Optional[Axes] = None
):
    """Helper function to plot Flux Rate summary for a list of GuiderSequences."""
    ax = ax if ax is not None else plt.gca()
    all_flux_rates, all_times = [], []
    for s in gseq:
        all_flux_rates.extend(s.get_flux_rates(sigmaclip_val=None))
        all_times.extend([f.ut_time for f in s.frames])

    ax.plot(all_times, all_flux_rates, "-", color="k", alpha=0.3)

    flux_rates = np.array([s.get_flux_rate_stats() for s in gseq])
    mid_times = get_mid_times(oseq.observations)
    xerr = np.array([timedelta(seconds=o.exptime / 2) for o in oseq])

    ax.errorbar(
        mid_times,
        flux_rates[:, 0],
        xerr=xerr,
        yerr=flux_rates[:, 1],
        fmt="o",
        capsize=5,
        label="Mean Flux Rate (no sigmaclip)",
        color="green",
    )

    ax.set_ylabel("Flux Rate (a.u.)")
    ax.set_title("Flux Rate of guide star fit")
    change_time_labels(ax, mid_times, oseq.time_range)
    ax.legend(loc="lower left")
    ymax = 1.1 * np.nanpercentile(all_flux_rates, 98)
    ax.set_ylim(0, ymax)


def _set_flux_rate_ylimits(ax: Axes, flux_rate_values: List[float]):
    """Set y-limits for Flux Rate plot with some padding."""
    if all(np.isnan(flux_rate_values)):
        ax.set_ylim(0, 1)
        return
    flux_rate_min = min(flux_rate_values)
    flux_rate_max = max(flux_rate_values)
    height = max(flux_rate_max - flux_rate_min, 0.1)
    y_padding = 0.2 * height
    ymin = max(0, flux_rate_min - y_padding)
    ymax = flux_rate_max + y_padding
    ax.set_ylim(ymin, ymax)


def plot_flux_rates_for_single_gseq(
    gseq: GuiderSequence,
    annotate_mean: bool = True,
    ax: Optional[plt.Axes] = None,
    **plot_kwargs,
) -> plt.Axes:
    """Plots the flux_rate as a function of time.

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
        The Axes object containing the flux_rate plot.
    """
    ax = plt.gca() if ax is None else ax

    flux_rates = gseq.get_flux_rates(sigmaclip_val=None)
    plot_kwargs.setdefault("marker", "x")
    plot_kwargs.setdefault("linestyle", "none")
    plot_kwargs.setdefault("markersize", 10)
    plot_kwargs.setdefault("color", "purple")
    ax.plot(gseq.guider_times, flux_rates, ls="-", color="gray", lw=0.5)
    ax.plot(gseq.guider_times, flux_rates, **plot_kwargs)
    _set_flux_rate_ylimits(ax, flux_rates)
    ax.set_xlabel("Time (UT)")
    ax.set_ylabel("Flux Rate (a.u.)")
    ax.set_title("Flux Rate")
    ax.grid(True)
    ts = gseq.observation.timeslot
    if ts is not None:
        change_time_labels(ax, gseq.guider_times, (ts.start_time, ts.end_time))

    if not annotate_mean:
        return ax
    mean_flux_rate = np.nanmean(flux_rates)
    std_flux_rate = np.nanstd(flux_rates)
    ax.axhline(mean_flux_rate, color="red", linestyle="--")
    ax.axhspan(
        mean_flux_rate - std_flux_rate,
        mean_flux_rate + std_flux_rate,
        color="red",
        alpha=0.2,
        label="$1\\sigma$ range",
    )
    ax.text(
        0.95,
        0.02,
        f"Mean Flux Rate: ${mean_flux_rate:.2f}\\pm {std_flux_rate:.2f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="red",
    )

    return ax
