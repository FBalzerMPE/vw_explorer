from matplotlib.axes import Axes
from typing import List, Tuple
from datetime import datetime

from matplotlib.ticker import MaxNLocator
from ..classes import Observation
import numpy as np

def change_time_labels(ax: Axes, mid_times: List[datetime], time_range: Tuple[datetime, datetime], fmt: str = "%H:%M:%S", max_num_labels: int = 6):
    """Change x-axis time labels to HH:MM:SS format, rotate them for better readability.
    Add extra ticks at start and end of time_range if only one mid_time is provided.

    """
    import matplotlib.dates as mdates

    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    for label in ax.get_xticklabels():
        label.set_rotation(40)
        label.set_horizontalalignment('right')

def get_mid_times(obs: List[Observation]) -> np.ndarray:
    """Get mid times for a list of Observation objects."""
    return np.array(
        [
            o.timeslot.mid_time if o.timeslot is not None else o.start_time_ut
            for o in obs
        ]
    )



def add_scale_bar(ax: Axes, pixel_scale: float = 0.53, length_arcsec: float =5, location: str ="lower left", color: str ="white", fontsize: int = 10):
    """
    Add a scale bar to the plot.
    """
    length_pixels = length_arcsec / pixel_scale

    if location == "lower left":
        x_start, y_start = 0.05, 0.05
    elif location == "lower right":
        x_start, y_start = 0.85, 0.05
    else:
        raise ValueError("Unsupported location. Use 'lower left' or 'lower right'.")

    x_data, y_data = ax.transAxes.transform((x_start, y_start))
    x_data, y_data = ax.transData.inverted().transform((x_data, y_data))

    ax.errorbar(x_data + length_pixels / 2, y_data, xerr=length_pixels / 2, fmt='none', ecolor=color, elinewidth=6, capsize=6, capthick=3)

    ax.text(
        x_data + length_pixels / 2,
        y_data + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
        f"${length_arcsec}$''",
        color="k",
        fontsize=fontsize,
        ha="center",
        va="bottom",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.08),

    )