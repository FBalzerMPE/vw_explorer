import argparse
from pathlib import Path

from vw_explorer.display.multi_file_plot import MultiFilePlotter
from vw_explorer.io import infer_vw_filenames, load_ifu_data
from vw_explorer.logger import LOGGER
from vw_explorer.plotting import plot_ifu_data


def parse_args():
    parser = argparse.ArgumentParser(description="Quicklook for VIRUS-W IFU images.")
    parser.add_argument(
        "fpaths",
        type=str,
        nargs="?",
        default=None,
        help="Input FITS file(s) in the format 'vwXXXXXX' or 'vwXXXXXX-YYY', or XXXX, or XXXX-YY where the hyphen indicates a range. Comma separators may also be used. If not provided, looks for the latest fits file following the format 'vw*.fits'.",
    )
    parser.add_argument(
        "cmap",
        type=str,
        nargs="?",
        default="hot",
        help="Colormap to use for the quicklook visualization.",
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    LOGGER.setLevel(args.loglevel.upper())
    filepaths = infer_vw_filenames(args.fpaths)
    LOGGER.info(f"Found {len(filepaths)} file(s).")

    def plot_ifu(fpath: Path):
        fiberpos, flux = load_ifu_data(fpath)
        plot_ifu_data(fiberpos, flux, "", cmap=args.cmap)

    MultiFilePlotter(filepaths, plot_ifu)


if __name__ == "__main__":
    main()
