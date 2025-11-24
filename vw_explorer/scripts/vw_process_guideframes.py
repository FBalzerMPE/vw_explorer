import argparse
from pathlib import Path
from typing import Optional

from vw_explorer import CONFIG
from vw_explorer.io.processing.data_processing import process_observation_data
from vw_explorer.io.processing.summary_plots import generate_dither_chunk_plots
from vw_explorer.logger import LOGGER



def parse_args():
    parser = argparse.ArgumentParser(
        description="Process observations taken during a VIRUS-W run, generating structured data based on the observation log and guider frames."
    )
    parser.add_argument(
        "-g",
        "--generate_dataframe",
        action="store_true",
        help="Whether to generate the observation DataFrame.",
    )
    parser.add_argument(
        "-f",
        "--force_guideframe_refit",
        action="store_true",
        help="Whether to force refitting (and replotting) the guide stars for their parameters in the guide frames.",
    )
    parser.add_argument(
        "-p",
        "--produce_plots",
        action="store_true",
        help="Whether to produce plots after processing.",
    )
    parser.add_argument(
        "--logfile_path",
        type=Path,
        help="Path to the observation log file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    filtered_chunks = None
    if args.generate_dataframe:
        logfile_path = CONFIG.sanitize_logfile_path(args.logfile_path)
        if logfile_path is None:
            raise ValueError("No log file found.")
        _, _, filtered_chunks = process_observation_data(
            logfile_path, force_log_reload=True, force_guide_refit=args.force_guideframe_refit
        )
    if args.produce_plots:
        output_dir = CONFIG.output_dir
        if filtered_chunks is None:
            LOGGER.info(
                "No prior processing done, thus the loading times will be longer."
            )
        generate_dither_chunk_plots(output_dir, filtered_chunks)
    if not args.generate_dataframe and not args.produce_plots:
        LOGGER.warning(
            "No action specified. Use --generate_dataframe and/or --produce_plots."
        )


if __name__ == "__main__":
    main()
