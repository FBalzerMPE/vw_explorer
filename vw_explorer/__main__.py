import argparse

from .io import create_guider_index, load_observations
from .logger import LOGGER


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create guider index and process observations."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    observations = load_observations()


if __name__ == "__main__":
    main()
