from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import pandas as pd

from ..constants import OUTPUT_PATH
from ..logger import LOGGER
from .log_parsing import parse_obs_logfile
from .dither_chunk_loading import load_dither_chunk_dataframe

if TYPE_CHECKING:
    from ..classes import Observation



def load_observations(
    logfile_path: Optional[Path] = None,
    backup_path=OUTPUT_PATH / "observations_raw.csv",
    force_log_reload: bool = False,
) -> List["Observation"]:
    """
    Loads observations from a logfile.
    If a backup CSV exists and reload_from_log is False, loads from the CSV instead.
    Otherwise, parses the logfile and saves a backup CSV.
    """
    from ..classes import Observation

    logfile_path = Path(logfile_path) if logfile_path else None
    if backup_path.exists() and not force_log_reload:
        obs_df = pd.read_csv(backup_path)
        df = Observation.from_dataframe(obs_df)
        LOGGER.info(
            f"Skipping log parsing, loading {len(df)} observations from backup CSV."
        )
        return df
    if logfile_path is None:
        raise ValueError(
            "Logfile path must be provided if no backup CSV exists or `force_log_reload` is True."
        )
    obs = parse_obs_logfile(logfile_path)
    obs_df = Observation.to_dataframe(obs)
    obs_df.to_csv(backup_path, index=False)
    LOGGER.info(f"Saved parsed {len(obs)} observations to backup CSV at {backup_path}.")
    load_dither_chunk_dataframe(observations=obs)
    return obs