import numpy as np
from typing import List, Optional
from typing_extensions import Literal

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ..calculations import get_clipping_kept_mask_by_distance


from ..classes import GuiderSequence


def _symmetrize_axis_limits(ax: Axes, min_width: Optional[float] = None):
    """Symmetrize the axis limits around the central position of the plot, using max range."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_center = (xlim[0] + xlim[1]) / 2
    y_center = (ylim[0] + ylim[1]) / 2
    x_range = max(abs(xlim[1] - x_center), abs(xlim[0] - x_center))
    y_range = max(abs(ylim[1] - y_center), abs(ylim[0] - y_center))
    x_range = min(max(x_range, y_range), 512)
    y_range = x_range
    if min_width is not None:
        x_range = max(x_range, min_width / 2)
        y_range = max(y_range, min_width / 2)
    xmin, xmax = x_center - x_range, x_center + x_range
    ymin, ymax = y_center - y_range, y_center + y_range
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

def _generate_centroid_legend(ax: Axes, **kwargs):
    """Generate a legend for centroid plots."""
    # Add legend with errorbar caps
    fiducial_marker = Line2D([], [], marker="X", color="blue", markersize=10, linestyle="None", label="Fid. coords from log")
    mean_marker = Line2D([], [], marker="o", color="red", markersize=8, linestyle="None", label="Mean position ($\\sigma$-clipped)")
    centroid_marker = Line2D([], [], marker="x", color="gray", alpha=0.5, linestyle="None", label="Individual GS fit positions")

    ax.legend(
        handles=[
            fiducial_marker,
            mean_marker,
            centroid_marker,
        ],
        **kwargs,
    )

def plot_centroid_series(
    gseqs: List[GuiderSequence],
    ax: Optional[Axes] = None,
    dithers: Optional[List[int]] = None,
):
    ax = ax if ax is not None else plt.gca()
    assert dithers is None or len(gseqs) == len(dithers), "Length of dithers must match number of GuiderSequences."
    for i, s in enumerate(gseqs):
        if len(s) == 0:
            continue
        color = plt.cm.tab10(i % 6)  # type: ignore
        s.plot_centroid_positions(
            "origin",
            ax=ax,
            set_limits=False,
            color=color,
            alpha=0.5,
            dither=dithers[i] if dithers else None,
        )
    _generate_centroid_legend(ax, loc="upper right", bbox_to_anchor=(-0.05, 1.01))
    _symmetrize_axis_limits(ax, 10)


def plot_centroids_for_single_gseq(
    gseq: GuiderSequence,
    relative_to: Literal["origin", "fiducial", "mean"] = "fiducial",
    ax: Optional[plt.Axes] = None,
    annotate_mean: bool = True,
    separate_outliers: bool = True,
    set_limits: bool = True,
    dither: Optional[int] = None,
    **scatter_kwargs,
) -> plt.Axes:
    """Scatter plot of fitted centroids from a GuiderSequence.

    Parameters
    ----------
    seq : GuiderSequence
        The guider sequence with fitted models.
    ax : plt.Axes, optional
        Matplotlib Axes to plot on. If None, creates a new figure and axes.
    scatter_kwargs : dict
        Additional keyword arguments passed to plt.scatter.

    Returns
    -------
    plt.Axes
        The Axes object containing the scatter plot.
    """
    ax = plt.gca() if ax is None else ax

    x_0: Optional[float] = None
    y_0: Optional[float] = None
    if relative_to == "fiducial":
        x_0, y_0 = gseq.observation.fiducial_coords
    elif relative_to == "mean":
        x_0, y_0 = gseq.get_centroid_stats()[0]
    centroids = gseq.get_centroids()
    x_centroids = centroids[:, 0]
    y_centroids = centroids[:, 1]
    title = "Sky pos."
    if x_0 is not None and y_0 is not None:
        x_centroids = x_centroids - x_0
        y_centroids = y_centroids - y_0
        title += f" (Rel. to ({x_0:.1f}, {y_0:.1f}))"

    scatter_kwargs.setdefault("s", 30)
    scatter_kwargs.setdefault("marker", "x")
    scatter_kwargs.setdefault("color", "k")
    if separate_outliers:
        clip_mask = get_clipping_kept_mask_by_distance(centroids)
        ax.scatter(x_centroids[clip_mask], y_centroids[clip_mask], **scatter_kwargs)
        ax.scatter(
            x_centroids[~clip_mask],
            y_centroids[~clip_mask],
            **{**scatter_kwargs, "color": "red", "alpha": 0.5},
        )
    else:
        ax.scatter(x_centroids, y_centroids, **scatter_kwargs)
    # Plot fiducial point
    x_fid, y_fid = 0, 0
    if relative_to == "origin":
        x_fid, y_fid = gseq.observation.fiducial_coords
    elif relative_to == "mean":
        x_fid, y_fid = gseq.observation.fiducial_coords - gseq.get_centroid_stats()[0]
    ax.plot(x_fid, y_fid, marker="X", color="blue", markersize=10, label="Fiducial")
    fid_str = ", ".join([str(round(c, 1)) for c in (gseq.observation.fiducial_coords)])
    ax.text(
        x_fid, y_fid, f"({fid_str})", color="blue", fontsize=12, ha="left", va="bottom"
    )
    ax.set_xlabel("X Centroid (pixels)")
    ax.set_ylabel("Y Centroid (pixels)")
    ax.grid(True)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")

    if set_limits:
        windowsize = 3
        if relative_to in ["fiducial", "mean"]:
            ax.set_xlim(-windowsize, windowsize)
            ax.set_ylim(-windowsize, windowsize)
            num_outside_box = np.sum(
                (np.abs(x_centroids) > windowsize) | (np.abs(y_centroids) > windowsize)
            )
        else:
            x_fid_0, y_fid_0 = gseq.observation.fiducial_coords
            ax.set_xlim(x_fid_0 - windowsize, x_fid_0 + windowsize)
            ax.set_ylim(y_fid_0 - windowsize, y_fid_0 + windowsize)
            num_outside_box = np.sum(
                (np.abs(x_centroids - x_fid_0) > windowsize)
                | (np.abs(y_centroids - y_fid_0) > windowsize)
            )
        if num_outside_box > 0:
            ax.text(
                0.95,
                0.05,
                f"{num_outside_box} points outside $\\pm{windowsize}$ px box",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                color="red",
            )
    if not annotate_mean:
        return ax
    (mean_x, mean_y), (std_x, std_y) = gseq.get_centroid_stats()
    if relative_to in ["fiducial", "mean"]:
        mean_x -= x_0
        mean_y -= y_0
    # Plot error bars for the mean centroid position
    ax.errorbar(
        mean_x,
        mean_y,
        xerr=std_x,
        yerr=std_y,
        fmt="o",
        color="red",
        ecolor="red",
        markersize=10,
        elinewidth=2,
        capsize=4,
        label="Mean ± Stddev",
    )
    _generate_centroid_legend(ax, loc="upper left", framealpha=0.2)
    if dither is not None:
        ax.text(
            mean_x,
            mean_y,
            f"D{dither}",
            # transform=ax.transAxes,
            ha="left",
            va="top",
            color="k",
        )
    return ax

