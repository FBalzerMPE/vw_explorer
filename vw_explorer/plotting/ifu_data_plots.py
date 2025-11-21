from typing import Optional
import matplotlib.cm as cm
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from matplotlib.colors import Normalize


def plot_ifu_data(
    fiberpos: np.ndarray,
    flux: np.ndarray,
    title: str,
    ax: Optional[Axes] = None,
    **kwargs,
):
    flux[flux <= 0.0] = 0.1
    flux = np.log(flux + 0.1)
    vmin, vmax = min(flux), max(flux)
    norm = Normalize(vmin=vmin, vmax=vmax)
    color = norm(flux)
    ax = ax if ax is not None else plt.gca()
    fig = plt.gcf()
    fig.suptitle(title, ha="center")
    kwargs.setdefault("marker", "h")
    kwargs.setdefault("s", 220.0)
    if "color" in kwargs:
        del kwargs["color"]
    c = ax.scatter(fiberpos[:, 1], fiberpos[:, 2], c=color, **kwargs)
    ax.set_xlabel("x [$''$]")
    ax.set_ylabel("y [$''$]")
    ax.axis("equal", adjustable="box")
    fig.colorbar(c, label="Normalized log(Flux + 0.1)")
    # Annotate min and max fiber flux
    s = f"Value range: {vmin:.2f} (min), {vmax:.2f} (max)"
    ax.text(0.05, 0.02, s, ha="left", va="bottom", transform=ax.transAxes)
    fig.set_size_inches(8, 6)
    ax.set_position([0.15, 0.15, 0.75, 0.75])  # type: ignore
