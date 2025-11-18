from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..classes import Observation
from ..constants import DATA_DIR
from ..logger import LOGGER


def _parse_obs_logfile(
    logfile_path: Path, base_datapath: Optional[Path]
) -> List[Observation]:
    """
    Parses a logfile and returns a list of Observations.

    """
    assert logfile_path.exists(), f"Logfile {logfile_path} does not exist."
    current_date = date.today()
    log_entries = []
    with logfile_path.open("r") as logfile:
        for i, line in enumerate(logfile):
            i += 1  # Line numbers start at 1
            if line.startswith("202"):
                date_str = line.strip()
                try:
                    current_date = date.fromisoformat(date_str)
                except Exception as e:
                    LOGGER.warning(
                        f"Could not parse date from line {i} ({e}): {line.strip()}"
                    )
                continue
            if not line.startswith("vw"):
                continue
            try:
                entry = Observation.from_line(
                    line, base_datapath=base_datapath, date=current_date
                )
                if entry:
                    log_entries.append(entry)
            except AssertionError as e:
                LOGGER.warning(f"Skipping malformed line {i} ({e}): {line.strip()}")
                continue
    return sorted(log_entries, key=lambda x: x.start_time_ut)


def load_observations(
    logfile_path: Optional[Path] = None,
    base_datapath: Optional[Path] = None,
    backup_path=DATA_DIR / "observations_raw.csv",
    reload_from_log: bool = False,
) -> List[Observation]:
    """
    Loads observations from a logfile.
    """
    logfile_path = Path(logfile_path) if logfile_path else None
    base_datapath = Path(base_datapath) if base_datapath else None
    if backup_path.exists() and not reload_from_log:
        obs_df = pd.read_csv(backup_path)
        return Observation.from_dataframe(obs_df, base_datapath=base_datapath)
    if logfile_path is None:
        raise ValueError(
            "Logfile path must be provided if no backup CSV exists or reload_from_log is True."
        )
    obs = _parse_obs_logfile(logfile_path, base_datapath)
    obs_df = Observation.to_dataframe(obs)
    obs_df.to_csv(backup_path, index=False)
    LOGGER.info(f"Saved parsed observations to backup CSV at {backup_path}.")
    return obs
