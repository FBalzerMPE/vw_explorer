import argparse
import logging
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import Optional
from typing import Tuple, List

import numpy as np
from astropy.io import fits
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.widgets import Button

_logger = logging.getLogger("vw_quicklook")


def _find_fits_list() -> List[Path]:
    fits_files = sorted(Path(".").glob("vw*.fits"), key=lambda x: x.stat().st_mtime)
    return fits_files


def _sanitize_fpath(f_in: Optional[str]) -> Path:
    if f_in is not None:
        return Path(f_in)
    else:
        fits_files = sorted(Path(".").glob("vw*.fits"), key=lambda x: x.stat().st_mtime)
        if not fits_files:
            _logger.error(
                "No FITS files found matching 'vw*.fits'. Please provide an input file."
            )
            raise FileNotFoundError("No FITS files following 'vw*.fits' format found.")
        return fits_files[-1]


def _get_fiberpos() -> np.ndarray:
    fpath = Path(__file__).parent / "vw_explorer/assets/IFUcen.txt"
    if not fpath.exists():
        raise FileNotFoundError(f"Fiber position file not found at {fpath}")
    fiberpos = np.loadtxt(fpath, comments="#")
    # Flip to match finderchart/guider
    fiberpos[:, 2] = fiberpos[:, 2] * -1.0
    return fiberpos


def plot_ifu_data(fiberpos: np.ndarray, flux: np.ndarray, title: str, cmap: str):
    flux[flux <= 0.0] = 0.1
    flux = np.log(flux + 0.1)
    color = (flux - min(flux)) / (max(flux) - min(flux))

    plt.figure(figsize=[6, 4])
    plt.figtext(0.5, 0.92, title, ha="center")
    plt.axes([0.15, 0.15, 0.75, 0.75], facecolor="w")  # type: ignore
    colors = cm.get_cmap(cmap)(color)  # type: ignore
    plt.scatter(fiberpos[:, 1], fiberpos[:, 2], 220.0, marker="h", color=colors)  # type: ignore
    plt.xlabel("x [$''$]")
    plt.ylabel("y [$''$]")
    plt.show()


def load_data(fits_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the FITS data and extracts fluxes for each fiber.
    """
    ycut = 2048  # y position to cut at
    xw, yw = 6.0, 300.0  # extraction box width
    kappa = 2.0  # Outlier rejection threshold

    _logger.info(f"Reading {fits_path}")
    fiberpos = _get_fiberpos()

    img: np.ndarray = fits.getdata(fits_path, ignore_missing_end=True)  # type: ignore
    if img.shape[0] < 4096:
        # assume y binned
        ycut /= 2
        yw /= 2
    if img.shape[1] < 2048:
        # assume x binned
        _logger.error("Unfortunately quicklook only works on non x binned images!")
        raise ValueError("Input image is x binned, which is not supported.")

    imA = img[0:, 0:1025]
    imA = imA - np.median(img[0:, 1030:1060])
    imB = img[0:, 1124:]
    imB = imB - np.median(img[0:, 1090:1120])

    img = np.concatenate((imA, imB), axis=1)

    flux = np.zeros(fiberpos.shape[0])

    # Collapse along fibers
    for i in range(fiberpos.shape[0]):
        xmin, xmax = int(fiberpos[i, 3] - xw / 2.0), int(fiberpos[i, 3] + xw / 2.0)
        ymin, ymax = int(ycut - yw / 2.0), int(ycut + yw / 2)
        f = (img[ymin:ymax, xmin:xmax]).flatten()

        mdflux = np.median(f[~np.isnan(f)])
        # Remove outliers
        f[abs(f - mdflux) > kappa * np.std(f)] = np.nan
        mnflux = np.mean(f[~np.isnan(f)])
        flux[i] = mnflux
        # img[ymin:ymax, xmin:xmax] = 0
    return fiberpos, flux


class QuickLookViewer:
    def __init__(self, files: List[Path], cmap: str = "gray"):
        if not files:
            raise FileNotFoundError("No FITS files to browse.")
        self.files = files
        self.cmap = cmap
        self.idx = 0

        self.fiberpos = None
        self.flux = None

        # create figure and axes
        self.fig = plt.figure(figsize=(7, 5))
        self.ax = self.fig.add_axes([0.08, 0.18, 0.84, 0.75])
        self.title_text = self.fig.text(0.5, 0.95, "", ha="center")

        # placeholders for scatter
        self.scat = None

        # buttons
        axprev = self.fig.add_axes([0.1, 0.03, 0.18, 0.06])
        axnext = self.fig.add_axes([0.32, 0.03, 0.18, 0.06])
        axopen = self.fig.add_axes([0.78, 0.03, 0.18, 0.06])

        self.bprev = Button(axprev, "Previous")
        self.bnext = Button(axnext, "Next")
        self.bopen = Button(axopen, "Open...")

        self.bprev.on_clicked(self.on_prev)
        self.bnext.on_clicked(self.on_next)
        self.bopen.on_clicked(self.on_open)

        # initial load
        self.load_index(self.idx)
        plt.show()

    def load_index(self, idx: int):
        path = self.files[idx]
        fiberpos, flux = load_data(path)
        self.fiberpos = fiberpos
        self.flux = flux
        self._draw(path.stem)

    def _draw(self, title: str):
        # prepare colors
        flux = self.flux.copy()
        flux[flux <= 0.0] = 0.1
        flux = np.log(flux + 0.1)
        color = (flux - min(flux)) / (max(flux) - min(flux))
        colors = cm.get_cmap(self.cmap)(color)

        self.ax.clear()
        self.ax.set_xlabel("x [$''$]")
        self.ax.set_ylabel("y [$''$]")
        self.scat = self.ax.scatter(
            self.fiberpos[:, 1], self.fiberpos[:, 2], 220.0, marker="h", color=colors
        )
        # if hasattr(self, "cbar"):
        #     self.cbar.remove()
        # self.cbar = self.fig.colorbar(self.scat, ax=self.ax, orientation="vertical")
        self.title_text.set_text(title)
        self.fig.canvas.draw_idle()

    def on_prev(self, event):
        if self.idx > 0:
            self.idx -= 1
            self.load_index(self.idx)

    def on_next(self, event):
        if self.idx < len(self.files) - 1:
            self.idx += 1
            self.load_index(self.idx)

    def on_open(self, event):
        # use tkinter dialog to pick a file
        root = tk.Tk()
        root.withdraw()
        fp = filedialog.askopenfilename(
            title="Open FITS file", filetypes=[("FITS", "*.fits"), ("All files", "*.*")]
        )
        root.destroy()
        if fp:
            p = Path(fp)
            # insert into list after current index and move to it
            self.files.insert(self.idx + 1, p)
            self.idx += 1
            self.load_index(self.idx)


def parse_args():
    parser = argparse.ArgumentParser(description="Quicklook for VIRUS-W IFU images.")
    parser.add_argument(
        "fpath",
        type=str,
        nargs="?",
        default=None,
        help="Input FITS file. If not provided, looks for the latest fits file following the format 'vw*.fits'.",
    )
    parser.add_argument(
        "cmap",
        type=str,
        nargs="?",
        default="gray",
        help="Colormap to use for the quicklook visualization.",
    )
    parser.add_argument(
        "--browse",
        action="store_true",
        help="Open the interactive browser with Prev/Next/Open buttons to load multiple files.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()

    if args.browse:
        files = _find_fits_list()
        if args.fpath:
            # put provided file first
            p = Path(args.fpath)
            if p.exists():
                files.insert(0, p)
        if not files:
            _logger.error("No FITS files found for browsing.")
            raise FileNotFoundError("No FITS files found.")
        QuickLookViewer(files, args.cmap)
        return

    fits_path = _sanitize_fpath(args.fpath)
    QuickLookViewer([fits_path], args.cmap)


if __name__ == "__main__":
    main()
