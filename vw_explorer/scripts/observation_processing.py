from pathlib import Path
from typing import Optional

from vw_explorer.io.processing.data_processing import process_observation_data
from vw_explorer.io.processing.summary_plots import generate_dither_chunk_plots
from vw_explorer import DATA_PATH
from vw_explorer.logger import LOGGER
import argparse

def _sanitize_logfile_path(input_path: Optional[Path]) -> Optional[Path]:
    logfile_path = DATA_PATH  if input_path is None else Path(input_path)
    if logfile_path.is_dir():
        logfile_path = logfile_path / "log.txt"
    if not logfile_path.exists():
        return None
    if input_path is None:
        LOGGER.info(f"No log file provided, using default at {logfile_path}")
    return logfile_path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Process observation data and generate plots."
    )
    parser.add_argument(
        "-g",
        "--generate_dataframe",
        action="store_true",
        help="Whether to generate the observation DataFrame.",
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
    if args.generate_dataframe:
        logfile_path = _sanitize_logfile_path(args.logfile_path)
        if logfile_path is None:
            raise ValueError("No log file found.")
        process_observation_data(logfile_path, force_log_reload=True)
    if args.produce_plots:
        output_dir = DATA_PATH / "output"
        generate_dither_chunk_plots(output_dir)
    if not args.generate_dataframe and not args.produce_plots:
        print("No action specified. Use --generate_dataframe and/or --produce_plots.")
if __name__ == "__main__":
    main()