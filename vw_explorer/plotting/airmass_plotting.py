from typing import List, Optional
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ..classes import ObservationSequence
from .util import change_time_labels, get_mid_times


def _set_am_limits(ax: Axes, am_values: List[float]):
    """Set y-limits for airmass plot with some padding."""
    if all(np.isnan(am_values)):
        ax.set_ylim(1.0, 2.0)
        return
    am_min = min(am_values)
    am_max = max(am_values)
    height = max(am_max - am_min, 0.1)
    y_padding = 0.2 * height
    ymin = max(1.0, min(am_min - y_padding, am_min * 0.95))
    ymax = max(am_max + y_padding, am_max * 1.05)
    ax.set_ylim(ymin, ymax)

def plot_airmass_series(
        oseq: ObservationSequence,
        ax: Optional[Axes] = None,
    ):
    ax = ax if ax is not None else plt.gca()
    am = [o.airmass for o in oseq]
    mid_times = get_mid_times(oseq.observations)
    ax.plot(mid_times, am, "o-", color="purple")
    
    ax.set_ylabel("Airmass")
    ax.set_title("Airmass")
    change_time_labels(ax, mid_times, oseq.time_range)
    _set_am_limits(ax, am)
    ax.text(0.02, 0.02, f"{am[0]:.2f}", va="bottom", ha="left", color="purple", transform=ax.transAxes)
    ax.text(0.98, 0.02, f"{am[-1]:.2f}", va="bottom", ha="right", color="purple", transform=ax.transAxes)

