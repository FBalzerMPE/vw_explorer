from matplotlib.axes import Axes
from typing import List, Tuple
from datetime import datetime
from ..classes import Observation
import numpy as np

def change_time_labels(ax: Axes, mid_times: List[datetime], time_range: Tuple[datetime, datetime], fmt: str = "%H:%M:%S"):
    """Change x-axis time labels to HH:MM:SS format, rotate them for better readability.
    Add extra ticks at start and end of time_range if only one mid_time is provided.

    """
    if len(mid_times) <= 1:
        mid_times = [time_range[0], mid_times[0], time_range[1]]
    t_labels = [st.strftime(fmt) for st in mid_times]
    ax.set_xticks(mid_times)
    ax.set_xticklabels(t_labels, rotation=45, ha="right")  

def get_mid_times(obs: List[Observation]) -> np.ndarray:
    """Get mid times for a list of Observation objects."""
    return np.array(
        [
            o.timeslot.mid_time if o.timeslot is not None else o.start_time_ut
            for o in obs
        ]
    )