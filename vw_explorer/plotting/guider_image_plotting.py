from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing_extensions import Literal
from ..classes import GuiderFrame, GuideStarModel, Observation


def plot_img_data(
    data: np.ndarray,
    ax: Optional[Axes] = None,
    add_cbar=False,
    cbar_size="4%",
    cbar_pad=0.05,
    **kwargs
):
    ax = plt.gca() if ax is None else ax
    vmin, vmax = np.percentile(data, [5, 95])
    vmin = kwargs.pop("vmin", vmin)
    vmax = kwargs.pop("vmax", vmax)
    kwargs["norm"] = kwargs.get("norm", LogNorm(vmin, vmax))
    kwargs["cmap"] = kwargs.get("cmap", "gray")
    # Full frame with marker
    im = ax.imshow(data, origin="lower", **kwargs)
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=cbar_size, pad=cbar_pad)
    if add_cbar:
        ax.get_figure().colorbar(im, cax=cax)
    else:
        # hide the reserved cax so it still consumes space but doesn't draw
        cax.set_visible(False)

def _prepare_frame_data(
    frame: GuiderFrame, mean_coords: Optional[Tuple[float, float]] = None, cutout_size: int = 20, 
) -> np.ndarray:
    if mean_coords is None:
        return frame.data
    return frame.get_cutout(*mean_coords, cutout_size)

def plot_frame_cutout(
    frame: GuiderFrame,
    mean_coords: Optional[Tuple[float, float]] = None,
    cutout_size: int = 20,
    ax: Optional[Axes] = None,
    **kwargs,
) -> Axes:
    """Plots data from the given guider frame in context of the observation.
    """
    ax = plt.gca() if ax is None else ax
    data = _prepare_frame_data(frame, mean_coords, cutout_size=cutout_size)
    plot_img_data(data, ax=ax, **kwargs)
    if mean_coords is not None:
        ax.plot(cutout_size // 2, cutout_size // 2, "bx", markersize=12)
        ax.text(cutout_size // 2 + 2, cutout_size // 2 + 2, f"{mean_coords[0]:.1f}, {mean_coords[1]:.1f}", color="b", bbox=dict(facecolor="white", alpha=0.7))
    return ax

def plot_guidefit_model(
    frame: GuiderFrame,
    model_fit: GuideStarModel,
) -> Figure:
    """Plots the guider frame data with the fitted 2D Gaussian model overlayed.

    Parameters
    ----------
    data : np.ndarray
        2D array of the guider frame data.
    fitted_model : StarModelFit
        The fitted 2D Gaussian model.
    """
    x_min, x_max, y_min, y_max = frame.get_cutout_coords(
        model_fit.x_cent_in, model_fit.y_cent_in, model_fit.size_in
    )
    rel_x_cent = model_fit.x_cent - x_min
    rel_y_cent = model_fit.y_cent - y_min
    cutout_data = model_fit.input_data

    y, x = np.mgrid[0 : cutout_data.shape[0], 0 : cutout_data.shape[1]]
    fitted_data = model_fit.model(x, y)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 4))
    vmin, vmax = np.percentile(cutout_data, [5, 99])
    # Original data
    plot_img_data(cutout_data, ax=ax1, vmin=vmin, vmax=vmax, add_cbar=True)
    ax1.set_title("Original Data")
    ax1.plot(rel_x_cent, rel_y_cent, "rx")
    # Fitted model
    plot_img_data(fitted_data, ax=ax2, vmin=vmin, vmax=vmax, add_cbar=False)
    ax2.plot(rel_x_cent, rel_y_cent, "rx")
    ax2.set_title("Fitted Model")
    # Residuals
    residuals = cutout_data - fitted_data
    m = np.nanmax(np.abs(residuals))
    try:
        from matplotlib.colors import TwoSlopeNorm

        norm = TwoSlopeNorm(vcenter=0.0, vmin=-m, vmax=m)
    except ImportError:
        from matplotlib.colors import Normalize

        norm = Normalize(vmin=-m, vmax=m)
    plot_img_data(residuals, ax=ax3, cmap="RdBu_r", norm=norm, add_cbar=True)
    ax3.set_title("Residuals")
    ax3.axis("off")
    plot_img_data(frame.data, ax=ax4, vmin=vmin, vmax=vmax)
    ax4.plot(model_fit.x_cent, model_fit.y_cent, "rx")
    ax4.set_title("Full Frame")
    ax4.plot([x_min, x_max], [y_min, y_min], color="yellow", linestyle="-")
    ax4.plot([x_min, x_max], [y_max, y_max], color="yellow", linestyle="-")
    ax4.plot([x_min, x_min], [y_min, y_max], color="yellow", linestyle="-")
    ax4.plot([x_max, x_max], [y_min, y_max], color="yellow", linestyle="-")
    ax4.axis("off")
    fig.tight_layout()
    return fig
