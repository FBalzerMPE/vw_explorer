from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import pandas as pd
from typing_extensions import Literal

from ..constants import CONFIG
from ..logger import LOGGER
from .log_parsing import parse_obs_logfile
from .dither_chunk_loading import load_dither_chunk_dataframe

if TYPE_CHECKING:
    from ..classes import Observation


def load_obs_dataframe(which: Literal["raw", "processed"] = "raw") -> pd.DataFrame:
    """
    Loads the observation DataFrame from a backup CSV file.

    Parameters
    ----------
    backup_path : Path
        Path to the backup CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing observation data.
    """
    backup_path = CONFIG.output_dir / f"observations_{which}.csv"
    obs_df = pd.read_csv(backup_path)
    LOGGER.debug(
        f"Loaded {len(obs_df)} observations from backup CSV at {backup_path}."
    )
    return obs_df

def load_observations(
    logfile_path: Optional[Path] = None,
    force_log_reload: bool = False,
) -> List["Observation"]:
    """
    Loads observations from a logfile.
    If a backup CSV exists and reload_from_log is False, loads from the CSV instead.
    Otherwise, parses the logfile and saves a backup CSV.
    """
    from ..classes import Observation

    logfile_path = CONFIG.sanitize_logfile_path(logfile_path)
    backup_path = CONFIG.output_dir / "observations_raw.csv"
    if backup_path.exists() and not force_log_reload:
        obs_df = load_obs_dataframe(which="raw")
        obs_list = Observation.from_dataframe(obs_df)
        LOGGER.debug(
            f"Skipping log parsing, loading {len(obs_list)} observations from backup CSV."
        )
        return obs_list
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