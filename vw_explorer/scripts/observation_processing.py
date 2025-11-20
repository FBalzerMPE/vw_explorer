from pathlib import Path

from vw_explorer.io.processing.data_processing import process_all_data
from vw_explorer import DATA_PATH
import argparse

def _sanitize_logfile_path(logfile_path: Path) -> Path:
    logfile_path = Path(logfile_path)
    if logfile_path.is_dir():
        logfile_path = logfile_path / "observation_log.csv"
    if not logfile_path.exists():
        logfile_path = DATA_PATH / logfile_path
    assert logfile_path.exists(), f"Log file {logfile_path} does not exist."
    return logfile_path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Process observation data and generate plots."
    )
    parser.add_argument(
        "logfile_path",
        type=Path,
        help="Path to the observation log file.",
    )
    return parser.parse_args()

def main(logfile_path: Path, force_log_reload: bool = True):
    logfile_path = _sanitize_logfile_path(logfile_path)
    process_all_data(logfile_path, force_log_reload)

if __name__ == "__main__":
    args = parse_args()
    main(args.logfile_path)