import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def plot_ifu_data(fiberpos: np.ndarray, flux: np.ndarray, title: str, **kwargs):
    flux[flux <= 0.0] = 0.1
    flux = np.log(flux + 0.1)
    color = (flux - min(flux)) / (max(flux) - min(flux))

    plt.figure(figsize=[6, 4])
    plt.figtext(0.5, 0.92, title, ha="center")
    plt.axes([0.15, 0.15, 0.75, 0.75], facecolor="w")  # type: ignore
    colors = cm.get_cmap(kwargs.pop("cmap", "hot"))(color)  # type: ignore
    kwargs.setdefault("markeredgecolor", "k")
    kwargs.setdefault("marker", "h")
    if "color" in kwargs:
        del kwargs["color"]
    c = plt.scatter(fiberpos[:, 1], fiberpos[:, 2], 220.0, marker=kwargs["marker"], color=colors, **kwargs)  # type: ignore
    plt.xlabel("x [$''$]")
    plt.ylabel("y [$''$]")
    plt.axis("equal")
    plt.colorbar(c, label="log(Flux + 0.1)")
