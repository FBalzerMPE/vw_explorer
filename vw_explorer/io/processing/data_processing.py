from pathlib import Path
from typing import List

import pandas as pd

from ...classes import GuiderSequence, Observation, DitherChunk
from ...constants import OUTPUT_PATH
from ...logger import LOGGER
from ..observation_loading import load_observations
from .plot_creation import generate_dither_chunk_plots
from ..dither_chunk_loading import load_dither_chunk_dataframe

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

def _get_dither_chunk_mapping(observations: List[Observation]) -> dict:
    """
    Creates a mapping from observation filenames to their dither chunk indices.

    Parameters
    ----------
    observations : List[Observation]
        List of Observation objects.

    Returns
    -------
    dict
        Mapping from observation filename to dither chunk index.
    """
    chunk_df = load_dither_chunk_dataframe()
    chunk_mapping = {}
    for obs in observations:
        if obs.is_calibration_obs:
            chunk_mapping[obs.filename] = -1
            continue
        sub_df = chunk_df[chunk_df["observation_names"].str.contains(obs.filename)]
        if not sub_df.empty:
            chunk_mapping[obs.filename] = sub_df.iloc[0]["chunk_index"]
            continue
        chunk_mapping[obs.filename] = -1
        LOGGER.warning(f"No dither chunk found for observation {obs.filename}.")
    return chunk_mapping

def process_all_data(logfile_path: Path, force_log_reload: bool = True) -> pd.DataFrame:
    observations = load_observations(
        logfile_path=logfile_path, force_log_reload=force_log_reload
    )
    obs_df = Observation.to_dataframe(observations)
    chunk_map = _get_dither_chunk_mapping(observations)
    obs_df["dither_chunk_index"] = obs_df["filename"].map(chunk_map)
    ch_dict = DitherChunk.get_all_dither_chunks(observations)
    chunks = [ch for ch_list in ch_dict.values() for ch in ch_list]
    LOGGER.info(f"Processing {len(chunks)} dither chunks from observations. Loading all guider sequences might take a while as we're fitting the guide stars.")
    guider_sequences = [g_seq for ch in chunks for g_seq in ch.obs_seq.get_guider_sequences()]
    generate_dither_chunk_plots(chunks, OUTPUT_PATH)
    seqs_df = GuiderSequence.get_combined_stats_df(guider_sequences)
    final_df = obs_df.merge(seqs_df, on="filename", how="left")
    output_file = OUTPUT_PATH / "observations_processed.csv"
    save_observations_to_csv(final_df, output_file)
    return final_df
