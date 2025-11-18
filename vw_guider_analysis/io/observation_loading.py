from collections import Counter
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import pandas as pd

from ..constants import CALIB_NAMES, DATA_DIR
from ..logger import LOGGER

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
            if line.startswith("202"):
                date_str = line.strip()
                try:
                    current_date = date.fromisoformat(date_str)
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
    return sorted(observations, key=lambda x: x.start_time_ut_noted)


def load_observations(
    logfile_path: Optional[Path] = None,
    base_datapath: Optional[Path] = None,
    backup_path=DATA_DIR / "observations_raw.csv",
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


def get_observation_summary(observations: List["Observation"]) -> str:
    """Provides a summary string for a list of observations."""
    earliest, latest = (
        min(obs.start_time_ut_noted for obs in observations),
        max(obs.start_time_ut_noted for obs in observations),
    )
    num_obs = len(observations)
    target_counts = Counter(obs.target for obs in observations)
    science_targets = {k: v for k, v in target_counts.items() if k not in CALIB_NAMES}
    science_targets = dict(sorted(science_targets.items(), key=lambda item: item[0]))
    calib_targets = {k: v for k, v in target_counts.items() if k in CALIB_NAMES}

    summary = (
        f"Observation Log Summary:\n"
        f"  Time Range:\n    {earliest} to\n    {latest}\n"
        f"  Total Observations: {num_obs}\n"
        f"  Target Counts for {len(science_targets)} unique targets:\n"
    )
    for target, count in science_targets.items():
        if target in CALIB_NAMES:
            continue
        summary += f"    {target + ':':<10} {count}\n"
    num_calibs = sum(calib_targets.values())
    summary += f"  Calibration Observations: {num_calibs}\n"
    return summary
