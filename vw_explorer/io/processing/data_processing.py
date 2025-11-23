from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from ...classes import DitherChunk, GuiderSequence, Observation
from ...constants import OUTPUT_PATH
from ...logger import LOGGER
from ..dither_chunk_loading import load_dither_chunk_dataframe
from ..observation_loading import load_observations


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
        sub_df = chunk_df[
            chunk_df["observation_names"].apply(lambda x: obs.filename in x)
        ]
        if not sub_df.empty:
            chunk_mapping[obs.filename] = sub_df.iloc[0]["chunk_index"]
            continue
        chunk_mapping[obs.filename] = -1
        LOGGER.warning(f"No dither chunk found for observation {obs.filename}.")
    return chunk_mapping


def process_observation_data(
    logfile_path: Path, force_log_reload: bool = True, force_guide_refit: bool = False
) -> Tuple[pd.DataFrame, List[DitherChunk], Optional[List[DitherChunk]]]:
    """Processes observation data from the log file and generates a DataFrame.
    Returns the DataFrame and list of DitherChunk objects which can be used
    for further processing as they contain the fitted guide star information.
    """
    observations = load_observations(
        logfile_path=logfile_path, force_log_reload=force_log_reload
    )
    obs_df = Observation.to_dataframe(observations)
    chunk_map = _get_dither_chunk_mapping(observations)
    obs_df["dither_chunk_index"] = obs_df["filename"].map(chunk_map)
    ch_dict = DitherChunk.get_all_dither_chunks(observations)
    chunks = [ch for ch_list in ch_dict.values() for ch in ch_list]
    relevant_chunks = [ch for ch in chunks if not ch.is_calibration_obs]

    output_fpath = OUTPUT_PATH / "observations_processed.csv"
    if output_fpath.exists() and not force_guide_refit:
        LOGGER.info(f"Loading existing processed data from {output_fpath}")
        existing_data = pd.read_csv(output_fpath)
        processed_filenames = set(existing_data["filename"])
        filtered_chunks = [
            ch
            for ch in relevant_chunks
            if not all(
                f.filename in processed_filenames for f in ch.obs_seq.observations
            )
        ]
        LOGGER.info(f"Found {len(filtered_chunks)} new dither chunks to process.")
    else:
        filtered_chunks = relevant_chunks
    num_frames = sum(len(ch.obs_seq) for ch in filtered_chunks)
    LOGGER.info(
        f"Found {len(chunks)} dither chunks, of which {len(relevant_chunks)} are from non-calibration observations.\nFitting guide stars for each of the {num_frames} frames amongst these might take a while."
    )
    guider_sequences = [
        g_seq
        for ch in tqdm(
            filtered_chunks, desc="Fitting guider sequences", colour="darkgreen"
        )
        for g_seq in ch.obs_seq.get_guider_sequences()
    ]
    seqs_df = GuiderSequence.get_combined_stats_df(guider_sequences)
    final_df = obs_df.merge(seqs_df, on="filename", how="left")
    if output_fpath.exists() and not force_guide_refit:
        final_df = (
            pd.concat([existing_data, final_df])
            .drop_duplicates(subset=["filename"])
            .reset_index(drop=True)
        )
    save_observations_to_csv(final_df.sort_values("target"), output_fpath)
    return final_df, chunks, filtered_chunks
