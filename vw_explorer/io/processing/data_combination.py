from pathlib import Path
from typing import List
import pandas as pd

from ...constants import OUTPUT_PATH
from ...logger import LOGGER
from ...classes import GuiderSequence, Observation
from .guider_sequence_loading import load_guider_sequences
from ..observation_loading import parse_or_load_observations


def save_observations_to_csv(df: pd.DataFrame, output_file: Path):
    """
    Saves the processed observations to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing processed observation data.
    output_file : Path
        Path to the output CSV file.
    """
    df.to_csv(output_file, index=False)
    LOGGER.info(f"Saved processed observations to {output_file}")


def combine_data(logfile_path: Path) -> pd.DataFrame:

    observations = parse_or_load_observations(
        logfile_path=logfile_path, force_log_reload=False
    )
    guider_sequences = load_guider_sequences(observations)
    obs_df = Observation.to_dataframe(observations)
    seqs_df = GuiderSequence.get_combined_stats_df(guider_sequences)
    final_df = obs_df.merge(seqs_df, on="filename", how="left")
    output_file = OUTPUT_PATH / "observations_processed.csv"
    save_observations_to_csv(final_df, output_file)
    return final_df
