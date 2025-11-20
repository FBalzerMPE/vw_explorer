from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

from ..classes import GuiderSequence, Observation, ObservationSequence
from ..logger import LOGGER


def _get_mid_times(obs: List[Observation]) -> np.ndarray:
    return np.array(
        [
            o.timeslot.mid_time if o.timeslot is not None else o.start_time_ut
            for o in obs
        ]
    )


def _plot_fwhm_sequence(
    gseq: List[GuiderSequence], oseq: ObservationSequence, ax: Optional[Axes] = None
):
    """Helper function to plot FWHM summary for a list of GuiderSequences."""
    ax = ax if ax is not None else plt.gca()
    all_fwhms, all_times = [], []
    for s in gseq:
        all_fwhms.extend(s.get_fwhms_arcsec(sigmaclip_val=None))
        all_times.extend([f.ut_time for f in s.frames])

    ax.plot(all_times, all_fwhms, "-", color="k", alpha=0.3)
    fwhms = np.array([s.get_fwhm_stats() for s in gseq])
    mid_times = _get_mid_times(oseq.observations)
    ax.errorbar(
        mid_times,
        fwhms[:, 0],
        yerr=fwhms[:, 1],
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
            color="red",
            label="Reported FWHM",
        )
    
    ax.set_ylabel("FWHM (arcsec)")
    ax.set_title("FWHM")
    t_labels = [st.strftime("%H:%M:%S") for st in mid_times]
    ax.set_xticks(mid_times, labels=t_labels, rotation=45, ha="right")
    ax.legend(loc="lower left")
    ymax = max(1.05 * np.nanmax(fwhms[:, 0]), 3)
    ax.set_ylim(0, ymax)


def _plot_amplitude_sequence(
    gseq: List[GuiderSequence], oseq: ObservationSequence, ax: Optional[Axes] = None
):
    """Helper function to plot Amplitude summary for a list of GuiderSequences."""
    ax = ax if ax is not None else plt.gca()
    all_amplitudes, all_times = [], []
    for s in gseq:
        all_amplitudes.extend(s.get_amplitudes(sigmaclip_val=None))
        all_times.extend([f.ut_time for f in s.frames])

    ax.plot(all_times, all_amplitudes, "-", label="All Fits", color="k", alpha=0.3)

    amplitudes = np.array([s.get_amplitude_stats() for s in gseq])
    mid_times = _get_mid_times(oseq.observations)

    ax.errorbar(
        mid_times,
        amplitudes[:, 0],
        yerr=amplitudes[:, 1],
        fmt="o",
        capsize=5,
        label="Fitted Amplitude",
        color="green",
    )
    
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title("Amplitude of guide star fit")
    t_labels = [st.strftime("%H:%M:%S") for st in mid_times]
    ax.set_xticks(mid_times, labels=t_labels, rotation=45, ha="right")
    ax.legend(loc="lower left")
    ymax = 1.1 * np.nanpercentile(all_amplitudes, 98)
    ax.set_ylim(0, ymax)


def _symmetrize_axis_limits(ax: Axes, min_width: Optional[float] = None):
    """Symmetrize the axis limits around the central position of the plot, using max range."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_center = (xlim[0] + xlim[1]) / 2
    y_center = (ylim[0] + ylim[1]) / 2
    x_range = max(abs(xlim[1] - x_center), abs(xlim[0] - x_center))
    y_range = max(abs(ylim[1] - y_center), abs(ylim[0] - y_center))
    if min_width is not None:
        x_range = max(x_range, min_width / 2)
        y_range = max(y_range, min_width / 2)
    ax.set_xlim(x_center - x_range, x_center + x_range)
    ax.set_ylim(y_center - y_range, y_center + y_range)


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
    _symmetrize_axis_limits(ax, 10)


def _plot_airmass_series(
        oseq: ObservationSequence,
        ax: Optional[Axes] = None,
    ):
    ax = ax if ax is not None else plt.gca()
    am = [o.airmass_noted for o in oseq]
    mid_times = _get_mid_times(oseq.observations)
    ax.plot(mid_times, am, "o-", color="purple")
    
    ax.set_ylabel("Airmass")
    ax.set_title("Airmass")
    t_labels = [st.strftime("%H:%M:%S") for st in mid_times]
    ax.set_xticks(mid_times, labels=t_labels, rotation=45, ha="right")  
    ax.set_ylim(0, 2)
    # Annotate start and stop
    print(am)
    ax.text(0.02, 0.2, f"{am[0]:.2f}", va="bottom", ha="left", color="purple", transform=ax.transAxes)
    ax.text(0.98, 0.2, f"{am[-1]:.2f}", va="bottom", ha="right", color="purple", transform=ax.transAxes)


def plot_guider_sequence_summary(oseq: ObservationSequence):
    summary = oseq.get_summary(max_line_length=40)
    gseq = oseq.get_guider_sequences(remove_failed=True)
    if gseq is None:
        LOGGER.error("Guider sequences not loaded. Call load_guider_sequences() first.")
        return
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1.5, 0.5, 1])
    ax1 = fig.add_subplot(gs[:2, 2:])  # Row 0, spans all columns
    ax2 = fig.add_subplot(gs[1, :2])  # Row 1, spans all columns
    ax3 = fig.add_subplot(gs[2, :2])  # Row 1, Column 0
    ax4 = fig.add_subplot(gs[2, 2:])  # Row 1, Column 1
    t = f"{oseq.sci_targets[0]} ({len(oseq)}), start: {oseq.observations[0].start_time_ut.strftime('%Y-%m-%d %H:%M')}"
    fig.suptitle(t, y=0.95)
    fig.text(
        0.04,
        0.9,
        summary,
        ha="left",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )
    dithers = [obs.dither for obs in oseq.observations]
    _plot_combined_centroids(gseq, ax1, dithers)
    _plot_airmass_series(oseq, ax2)
    _plot_fwhm_sequence(gseq, oseq, ax3)
    _plot_amplitude_sequence(gseq, oseq, ax4)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=0.1)  # type: ignore
