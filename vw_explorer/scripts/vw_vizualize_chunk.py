"""This is deprecated for now."""

import argparse
from typing import Tuple

from vw_explorer.classes import DitherChunk
from vw_explorer.display.multi_file_plot import MultiFilePlotter
from vw_explorer.io import infer_vw_filenames
from vw_explorer.logger import LOGGER


def parse_args():
    parser = argparse.ArgumentParser(description="Quicklook for VIRUS-W IFU images.")
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="The name of the target object to display.",
    )
    parser.add_argument(
        "--dither_chunk_index",
        type=int,
        default=0,
        help="The dither chunk index to visualize.",
    )
    parser.add_argument(
        "--fiducial_coords",
        type=str,
        help="Fiducial coordinates for the guider in the format 'x,y' (e.g., '512.1,512').",
    )
    parser.add_argument(
        "--fpaths",
        type=str,
        nargs="?",
        default=None,
        help="Input FITS file(s) in the format 'vwXXXXXX' or 'vwXXXXXX-YYY', or XXXX, or XXXX-YY where the hyphen indicates a range. If not provided, looks for the latest fits file following the format 'vw*.fits'.",
    )
    parser.add_argument(
        "cmap",
        type=str,
        nargs="?",
        default="gray",
        help="Colormap to use for the quicklook visualization.",
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    return parser.parse_args()


def _parse_fiducial_coords(coord_str: str) -> Tuple[float, float]:
    try:
        x_str, y_str = coord_str.split(",")
        x, y = float(x_str.strip()), float(y_str.strip())
        return x, y
    except Exception as e:
        raise ValueError(
            f"Invalid fiducial coordinates format: '{coord_str}'. Expected format is 'x,y'."
        ) from e


def main():
    args = parse_args()
    LOGGER.setLevel(args.loglevel.upper())
    filepaths = infer_vw_filenames(args.fpaths)
    LOGGER.info(f"Found {len(filepaths)} file(s).")
    fid_x, fid_y = _parse_fiducial_coords(args.fiducial_coords)
    seq = DitherChunk.from_filenames(filepaths)
    for obs in seq.observations:
        obs.fiducial_coords = (fid_x, fid_y)
    LOGGER.info(f"Observation Sequence Summary:\n{seq}")
    import matplotlib.pyplot as plt

    plt.show()


if __name__ == "__main__":
    main()
