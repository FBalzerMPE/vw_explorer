from matplotlib.axes import Axes
from typing import List, Tuple
from datetime import datetime
from ..classes import Observation
import numpy as np

def change_time_labels(ax: Axes, mid_times: List[datetime], time_range: Tuple[datetime, datetime], fmt: str = "%H:%M:%S", max_num_labels: int = 6):
    """Change x-axis time labels to HH:MM:SS format, rotate them for better readability.
    Add extra ticks at start and end of time_range if only one mid_time is provided.

    """
    if len(mid_times) <= 1:
        mid_times = [time_range[0], mid_times[0], time_range[1]]
    elif len(mid_times) > max_num_labels:
        # Instead, construct evenly spaced times within the time range
        new_times = []
        for i in range(max_num_labels):
            frac = i / (max_num_labels - 1)
            new_time = time_range[0] + (time_range[1] - time_range[0]) * frac
            new_times.append(new_time)
        mid_times = new_times
    t_labels = [st.strftime(fmt) for st in mid_times]
    ax.set_xticks(mid_times)
    ax.set_xticklabels(t_labels, rotation=40, ha="right")  

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