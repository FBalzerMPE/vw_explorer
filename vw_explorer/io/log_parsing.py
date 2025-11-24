from collections import Counter
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, List

from ..logger import LOGGER
from ..constants import CONFIG
from .log_sanitization import filter_and_clean_logfile, parse_date_line

if TYPE_CHECKING:
    from ..classes import Observation


def parse_obs_logfile(logfile_path: Path) -> List["Observation"]:
    """
    Parses a logfile and returns a list of Observations.

    """
    # Walk the base datapath to find the file
    avail_files = {f.stem: f for f in CONFIG.obs_dir.glob("**/vw*.fits")}
    from ..classes import Observation

    current_date = date.today()
    log_data = filter_and_clean_logfile(logfile_path)
    observations: List[Observation] = []
    for line_number, line in log_data.items():
        if line.startswith("# date: "):
            current_date = parse_date_line(line, line_number=line_number)
            continue
        try:
            entries = Observation.parse_obs_log_line(line, date=current_date, avail_files=avail_files)
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
    assert len(observations) > 0, "No observations were parsed from the logfile."
    cts = Counter(obs.filename for obs in observations)
    assert all(
        count == 1 for count in cts.values()
    ), f"Parsed observations contain duplicate filenames: {[filename for filename, count in cts.items() if count > 1]}"

    return sorted(observations, key=lambda x: x.start_time_ut)
