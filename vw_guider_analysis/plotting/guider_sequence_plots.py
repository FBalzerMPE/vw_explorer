import os
from pathlib import Path
from typing import Iterable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

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


def scatter_positions(
    seq: GuiderSequence,
    x_0: Optional[float] = None,
    y_0: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
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

    centroids = seq.get_centroids()
    x_cent = centroids[:, 0]
    y_cent = centroids[:, 1]
    title = "Fitted Centroid Positions"
    if x_0 is not None and y_0 is not None:
        x_cent = x_cent - x_0
        y_cent = y_cent - y_0
        title += f" (Relative to ({x_0:.1f}, {y_0:.1f}))"

    scatter_kwargs.setdefault("s", 30)
    scatter_kwargs.setdefault("marker", "x")
    scatter_kwargs.setdefault("color", "k")
    ax.scatter(x_cent, y_cent, **scatter_kwargs)
    ax.set_xlabel("X Centroid (pixels)")
    ax.set_ylabel("Y Centroid (pixels)")
    ax.grid(True)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")

    return ax


def plot_fwhm_sequence(
    seq: GuiderSequence,
    ax: Optional[plt.Axes] = None,
    **plot_kwargs,
) -> plt.Axes:
    """Plot FWHM (arcsec) over frame index for a GuiderSequence.

    Parameters
    ----------
    seq : GuiderSequence
        The guider sequence with fitted models.
    ax : plt.Axes, optional
        Matplotlib Axes to plot on. If None, creates a new figure and axes.
    plot_kwargs : dict
        Additional keyword arguments passed to plt.plot.

    Returns
    -------
    plt.Axes
        The Axes object containing the FWHM plot.
    """
    ax = plt.gca() if ax is None else ax

    plot_kwargs.setdefault("marker", "x")
    plot_kwargs.setdefault("linestyle", "--")
    plot_kwargs.setdefault("lw", 0.5)
    plot_kwargs.setdefault("markerfacecolor", "k")
    ax.plot(seq.guider_times, seq.fwhms_arcsec, **plot_kwargs)
    ax.set_xlabel("Time (UT)")
    ax.set_ylabel("FWHM (arcsec)")
    ax.set_title("FWHM over Guider Sequence Frames")
    ax.grid(True)

    return ax
