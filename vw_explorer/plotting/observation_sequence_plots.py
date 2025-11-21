import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from vw_explorer.classes.dither_chunk import DitherChunk

from ..logger import LOGGER
from .amplitude_plotting import plot_amplitude_series
from .fwhm_plotting import plot_fwhm_series
from .airmass_plotting import plot_airmass_series
from .centroid_plotting import plot_centroid_series


def plot_dither_chunk_summary(dchunk: DitherChunk):
    """Plots a summary of the guider sequences within a dither chunk.
    """
    oseq = dchunk.obs_seq
    summary = oseq.get_summary(max_line_length=40)
    gseq = oseq.get_guider_sequences(remove_failed=True)
    if len(gseq) == 0:
        LOGGER.warning(
            f"Target '{dchunk.target}', DC{dchunk.chunk_index}: No guider sequences found, skipping plot."
        )
        return
    if gseq is None:
        LOGGER.error("Guider sequences not loaded. Call load_guider_sequences() first.")
        return
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 4, height_ratios=[1.5, 0.5, 1])
    ax1 = fig.add_subplot(gs[:2, 2:])  # Row 0, spans all columns
    ax2 = fig.add_subplot(gs[1, :2])  # Row 1, spans all columns
    ax3 = fig.add_subplot(gs[2, :2])  # Row 1, Column 0
    ax4 = fig.add_subplot(gs[2, 2:])  # Row 1, Column 1
    t = f"{dchunk.target} [DC {dchunk.chunk_index}] ({len(oseq)}), start: {dchunk.time_range[0].strftime('%Y-%m-%d %H:%M')}"
    fig.suptitle(t, y=0.95, ha="left", x=0.05, fontsize=16)
    fig.text(
        0.04,
        0.9,
        summary,
        ha="left",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )
    dithers = [obs.dither for obs in oseq.observations]
    plot_centroid_series(gseq, ax1, dithers)
    plot_airmass_series(oseq, ax2)
    plot_fwhm_series(gseq, oseq, ax3)
    plot_amplitude_series(gseq, oseq, ax4)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=0.1)  # type: ignore
