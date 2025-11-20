from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, List

from ..logger import LOGGER
from .log_sanitization import filter_and_clean_logfile, parse_date_line

if TYPE_CHECKING:
    from ..classes import Observation


def parse_obs_logfile(logfile_path: Path) -> List["Observation"]:
    """
    Parses a logfile and returns a list of Observations.

    """
    from ..classes import Observation

    current_date = date.today()
    log_data = filter_and_clean_logfile(logfile_path)
    observations = []
    for line_number, line in log_data.items():
        if line.startswith("# date: "):
            current_date = parse_date_line(line, line_number=line_number)
            continue
        try:
            entries = Observation.parse_obs_log_line(line, date=current_date)
            if not entries:
                continue
            if len(entries) > 20:
                msg = f"Line {line_number} yielded {len(entries)} observations, which seems excessive. Possible parsing or log error, maybe check the log there."
                LOGGER.warning(msg)
            observations.extend(entries)
        except AssertionError as e:
            LOGGER.error(
                f"Could not parse line {line_number}:\n\t[{e}]\n\t{line.strip()}"
            )
            continue
    return sorted(observations, key=lambda x: x.start_time_ut)
