from typing import List, Optional, TYPE_CHECKING
import pandas as pd

from ..constants import OUTPUT_PATH
from ..logger import LOGGER

if TYPE_CHECKING:
    from ..classes import DitherChunk, Observation

def load_dither_chunk(target_name: str, chunk_index: int 
) -> "DitherChunk":
    """
    Loads a specific dither chunk from a DataFrame.
    Note that information on Observations is ONLY loaded from
    the fits files!

    Parameters
    ----------
    chunks_df : pd.DataFrame
        DataFrame containing dither chunk information.
    chunk_index : int
        The index of the dither chunk to load.
    target_name : str
        The name of the target for the dither chunk.

    Returns
    -------
    DitherChunk
        The requested DitherChunk object.
    """
    from ..classes import DitherChunk
    from .observation_loading import load_obs_df

    chunks_df = load_dither_chunk_dataframe()
    chunk_series = chunks_df[
        (chunks_df["chunk_index"] == chunk_index)
        & (chunks_df["target"] == target_name)
    ]
    if chunk_series.empty:
        avail_targets = chunks_df["target"].unique()
        raise ValueError(
            f"No dither chunk with index {chunk_index} found for target '{target_name}'. Available targets are:\n{avail_targets}"
        )
    try:
        obs_df = load_obs_df()
    except Exception as e:
        obs_df = None
        LOGGER.error(f"Failed to load observations: {e}, will load them from fits files instead.")
    # Convert the Series to a DitherChunk
    return DitherChunk.from_series(chunk_series.iloc[0], obs_df=obs_df)

def load_dither_chunk_dataframe(backup_fpath=OUTPUT_PATH / "dither_chunks.csv", observations: Optional[List["Observation"]] = None) -> pd.DataFrame:
    """
    Creates a backup CSV file containing dither chunk information for the given observations.

    Parameters
    ----------
    backup_fpath : Path
        Path to the backup CSV file.
    observations : Optional[List[Observation]]
        List of Observation objects to process. If None, attempts to load from backup CSV.
    """
    from ..classes import DitherChunk

    # Generate dither chunks and save to backup
    if backup_fpath.exists() and observations is None:
        chunks_df = pd.read_csv(backup_fpath)
        LOGGER.debug(
            f"Skipping dither chunk generation, loading chunks from backup CSV at {backup_fpath}."
        )
        # parse list columns:
        for col in ["observation_names", "observation_paths"]:
            chunks_df[col] = chunks_df[col].apply(
                    lambda x: [item.strip().strip("'") for item in x.strip("[]").split(",")]
                )
        return chunks_df
    if observations is None:
        raise ValueError(
            "Observations must be provided if no backup CSV exists."
        )
    chunks = DitherChunk.get_all_dither_chunks(observations)
    chunks_flat = [
        chunk
        for target_chunks in chunks.values()
        for chunk in target_chunks
    ]
    chunks_df = DitherChunk.to_dataframe(chunks_flat)
    chunks_df.to_csv(backup_fpath, index=False)
    LOGGER.info(
        f"Saved {len(chunks_flat)} dither chunks to backup CSV at {backup_fpath}."
        )
    return chunks_df