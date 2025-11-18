import argparse

from .io import load_observations, process_observations, create_guider_index
from .logger import LOGGER


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create guider index and process observations."
    )
    parser.add_argument(
        "--guider-dir",
        type=str,
        default=None,
        help="Path to the guider directory to find guider frames and create the index in.",
    )
    parser.add_argument(
        "--logfile-path",
        type=str,
        default=None,
        help="Path to the observation log file to load observations from.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    observations = load_observations(logfile_path=args.logfile_path)
    LOGGER.info(f"Loaded {len(observations)} observations.")
    create_guider_index(guider_dir=args.guider_dir, silent=False)
    process_observations(observations)


if __name__ == "__main__":
    main()
