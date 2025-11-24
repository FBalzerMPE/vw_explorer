

from matplotlib.axes import Axes
from ..classes import GuiderSequence, ObservationSequence
from typing import List, Optional
import matplotlib.pyplot as plt
from .util import get_mid_times

def plot_guide_frame_nums(
        gseqs: List[GuiderSequence],
        oseq: ObservationSequence,
        ax: Optional[Axes] = None,
):
    """Helper function to plot a bar chart of the amount of guide frames per GuiderSequence with respect to time."""
    ax = ax if ax is not None else plt.gca()
    all_nums = [len(s.frames) for s in gseqs]
    all_times = get_mid_times(oseq.observations)
    widths = [s.observation.exptime / 86400 for s in gseqs]

    bars = ax.bar(all_times, all_nums, width=widths, align='center', color='orange', edgecolor='black', alpha=0.2, lw=3)
    # ax.set_ylabel("$N_{\\rm frames}$")
    
    for bar, num, gs in zip(bars, all_nums, gseqs):
        dither = gs.observation.dither
        text = f"D{dither}: {num}"
        height = bar.get_height()
        height_offset = ax.get_ylim()[1] * 0.01
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + height_offset,
            text,
            ha='center', va='bottom', fontsize=10, color='black'
        )

    # Adjust y-limits to ensure annotations are visible
    ax.set_ylim(0, max(all_nums) * 1.4)
