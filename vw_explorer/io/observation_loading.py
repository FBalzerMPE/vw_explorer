from collections import Counter
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import pandas as pd

from ..constants import CALIB_NAMES, DATA_PATH
from ..logger import LOGGER
from ..util import parse_isoformat

if TYPE_CHECKING:
    from ..classes import Observation


def _parse_obs_logfile(
    logfile_path: Path, base_datapath: Optional[Path]
) -> List["Observation"]:
    """
    Parses a logfile and returns a list of Observations.

    """
    from ..classes import Observation

    assert logfile_path.exists(), f"Logfile {logfile_path} does not exist."
    current_date = date.today()
    observations = []
    with logfile_path.open("r") as logfile:
        for i, line in enumerate(logfile):
            i += 1  # Line numbers start at 1
            if line.startswith("# date: "):
                date_str = line.strip()[7:]
                try:
                    current_date = parse_isoformat(date_str + "T00:00:00").date()
                except Exception as e:
                    LOGGER.warning(
                        f"Could not parse date from line {i}:\n\t[{e}]\n\t{line.strip()}"
                    )
                continue
            if not line.startswith("vw"):
                continue
            try:
                entries = Observation.parse_obs_log_line(
                    line, base_datapath=base_datapath, date=current_date
                )
                if entries:
                    if len(entries) > 20:
                        msg = f"Line {i} yielded {len(entries)} observations, which seems excessive. Possible parsing or log error, maybe check the log there."
                        LOGGER.warning(msg)
                    observations.extend(entries)
            except AssertionError as e:
                LOGGER.error(f"Could not parse line {i}:\n\t[{e}]\n\t{line.strip()}")
                continue
    return sorted(observations, key=lambda x: x.start_time_ut)


def load_observations(
    logfile_path: Optional[Path] = None,
    base_datapath: Optional[Path] = DATA_PATH / "observations",
    backup_path=DATA_PATH / "observations_raw.csv",
    reload_from_log: bool = False,
) -> List["Observation"]:
    """
    Loads observations from a logfile.
    If a backup CSV exists and reload_from_log is False, loads from the CSV instead.
    Otherwise, parses the logfile and saves a backup CSV.
    """
    from ..classes import Observation

    logfile_path = Path(logfile_path) if logfile_path else None
    base_datapath = Path(base_datapath) if base_datapath else None
    if backup_path.exists() and not reload_from_log:
        obs_df = pd.read_csv(backup_path)
        df = Observation.from_dataframe(obs_df)
        LOGGER.info(
            f"Skipping log parsing, loading {len(df)} observations from backup CSV."
        )
        return df
    if logfile_path is None:
        raise ValueError(
            "Logfile path must be provided if no backup CSV exists or reload_from_log is True."
        )
    obs = _parse_obs_logfile(logfile_path, base_datapath)
    obs_df = Observation.to_dataframe(obs)
    obs_df.to_csv(backup_path, index=False)
    LOGGER.info(f"Saved parsed {len(obs)} observations to backup CSV at {backup_path}.")
    return obs
