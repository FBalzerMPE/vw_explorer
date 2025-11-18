import os
from pathlib import Path
from typing import Iterable, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from ..calculations import get_clipped_mask, get_clipped_mask_by_distance
from ..classes.guider_sequence import GuiderSequence
from .guider_frame_plots import plot_guidefit_model


def _fig_to_rgb_array(fig: Figure) -> np.ndarray:
    """Render fig to an (H, W, 3) uint8 RGB array."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    if hasattr(fig.canvas, "tostring_rgb"):
        buf = fig.canvas.tostring_rgb()  # type: ignore
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))
    else:
        buf = fig.canvas.buffer_rgba()  # type: ignore
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))[:, :, :3]
    # arr = buf.reshape((h, w, 3))
    return arr


def create_guider_gif(
    seq: GuiderSequence,
    out_path: Union[Path, str],
    fps: int = 5,
    figsize: tuple[float, float] = (12, 4),
    dpi: int = 100,
    frames: Optional[Iterable[int]] = None,
    close_fig: bool = True,
):
    """
    Make a GIF from a GuiderSequence using plot_guidefit_model for each frame.
    - seq: GuiderSequence instance (seq.frames and seq.models must be populated).
    - out_path: path to write .gif
    - fps: frames per second
    - figsize / dpi: ensure identical frame sizes
    - frames: optional iterable of indices to include (default: all)
    """
    out_path = Path(out_path)
    if frames is None:
        indices = range(min(len(seq.frames), len(seq.models)))
    else:
        indices = list(frames)

    imgs = []
    for i in indices:
        frame = seq.frames[i]
        model = seq.models[i]
        # ensure the plotting function returns a Figure (as implemented)
        fig = plot_guidefit_model(frame, model)
        fig.set_size_inches(*figsize)
        fig.set_dpi(dpi)
        fig.suptitle(f"Frame {i}", fontsize=16)
        img = _fig_to_rgb_array(fig)
        imgs.append(img)
        if close_fig:
            plt.close(fig)

    pil_imgs = [Image.fromarray(im) for im in imgs]
    duration = int(1000 / fps)
    pil_imgs[0].save(
        str(out_path),
        save_all=True,
        append_images=pil_imgs[1:],
        duration=duration,
        loop=0,
    )
    return out_path


def plot_centroid_positions(
    gseq: GuiderSequence,
    relative_to: Literal["origin", "fiducial", "mean"] = "fiducial",
    ax: Optional[plt.Axes] = None,
    annotate_mean: bool = True,
    separate_outliers: bool = True,
    set_limits: bool = True,
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
    title = "Fitted Centroid Positions"
    if x_0 is not None and y_0 is not None:
        x_centroids = x_centroids - x_0
        y_centroids = y_centroids - y_0
        title += f" (Relative to ({x_0:.1f}, {y_0:.1f}))"

    scatter_kwargs.setdefault("s", 30)
    scatter_kwargs.setdefault("marker", "x")
    scatter_kwargs.setdefault("color", "k")
    if separate_outliers:
        clip_mask = get_clipped_mask_by_distance(centroids)
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
    return ax


def plot_fwhm_sequence(
    gseq: GuiderSequence,
    ax: Optional[plt.Axes] = None,
    annotate_mean: bool = True,
    **plot_kwargs,
) -> plt.Axes:
    """Plot FWHM (arcsec) over frame index for a GuiderSequence.

    Parameters
    ----------
    seq : GuiderSequence
        The guider sequence with fitted models.
    ax : plt.Axes, optional
        Matplotlib Axes to plot on. If None, creates a new figure and axes.
    annotate_mean : bool, optional
        Whether to annotate the mean and stddev on the plot. Default is True.
    plot_kwargs : dict
        Additional keyword arguments passed to plt.plot.

    Returns
    -------
    plt.Axes
        The Axes object containing the FWHM plot.
    """
    ax = plt.gca() if ax is None else ax
    plot_kwargs = plot_kwargs.copy()

    mean_fwhm, std_fwhm = gseq.get_fwhm_stats()
    plot_kwargs.setdefault("marker", "x")
    plot_kwargs.setdefault("linestyle", "--")
    plot_kwargs.setdefault("lw", 0.5)
    plot_kwargs.setdefault("markerfacecolor", "k")
    all_fwhms = gseq.get_fwhms_arcsec(sigmaclip_val=None)
    clip_mask = get_clipped_mask(all_fwhms)
    ax.plot(gseq.guider_times[clip_mask], all_fwhms[clip_mask], **plot_kwargs)
    plot_kwargs["alpha"] = 0.5
    plot_kwargs["color"] = "red"
    plot_kwargs["linestyle"] = "None"
    plot_kwargs["marker"] = "x"
    plot_kwargs["markersize"] = 8
    ax.plot(gseq.guider_times[~clip_mask], all_fwhms[~clip_mask], **plot_kwargs)
    ymax = max(1.05 * np.nanmax(all_fwhms), 3)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Time (UT)")
    ax.set_ylabel("FWHM (arcsec)")
    ax.set_title("FWHM over Guider Sequence Frames")
    ax.grid(True)
    if not annotate_mean:
        return ax
    ax.axhline(mean_fwhm, color="red", linestyle="--")
    ax.axhspan(
        mean_fwhm - std_fwhm,
        mean_fwhm + std_fwhm,
        color="red",
        alpha=0.2,
        label="$1\\sigma$ range",
    )
    ax.text(
        0.95,
        0.95,
        f"Mean FWHM: ${mean_fwhm:.2f}\\pm {std_fwhm:.2f}$ arcsec",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="red",
    )

    return ax
