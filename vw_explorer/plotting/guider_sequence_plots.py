from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image
from ..classes.guider_sequence import GuiderSequence
from .guider_image_plotting import plot_guidefit_model


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
    figsize: Tuple[float, float] = (12, 4),
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

def plot_guider_sequence_summary(gseq: GuiderSequence) -> Figure:
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), gridspec_kw={"height_ratios": [2, 1]})
    gseq.plot_initial_frame(ax=axes[0, 0], center_around="fiducial", cutout_size=70)
    gseq.plot_centroid_positions("fiducial", ax=axes[0, 1])
    gseq.plot_fwhm_timeseries(ax=axes[1, 0])
    gseq.plot_amplitude_timeseries(ax=axes[1, 1])
    return fig