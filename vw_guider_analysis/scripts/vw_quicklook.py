import argparse

import vw_guider_analysis as vwg


def parse_args():
    parser = argparse.ArgumentParser(description="Quicklook for VIRUS-W IFU images.")
    parser.add_argument(
        "fpath",
        type=str,
        nargs="?",
        default=None,
        help="Input FITS file(s). If not provided, looks for the latest fits file following the format 'vw*.fits'.",
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


def main():
    args = parse_args()
    vwg.LOGGER.setLevel(args.loglevel.upper())
    fits_path = vwg.sanitize_obs_fpath(args.fpath)
    fiberpos, flux = vwg.io.load_ifu_data.load_ifu_data(fits_path)
    vwg.plotting.ifu_data_plots.plot_ifu_data(
        fiberpos, flux, fits_path.stem, cmap=args.cmap
    )
    import matplotlib.pyplot as plt

    plt.show()


if __name__ == "__main__":
    main()
