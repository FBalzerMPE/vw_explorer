from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from ..logger import LOGGER
from ..util import parse_isoformat


def _check_date_order(lines: List[str]) -> int:
    """
    Checks whether the list of dates is in a sensible order.
    For this, we check that they generally decrease or at most increase by one.
    """
    dates = [
        parse_date_line(l, i)
        for i, l in enumerate(lines, start=1)
        if l.startswith("# date: ")
    ]
    if len(dates) == 0:
        LOGGER.warning(
            "No date lines found in logfile.\nPlease add them (fmt: # date: YYYY-MM-DD) to ensure correct parsing."
        )
        raise ValueError("No date lines found in logfile.")
    for i in range(1, len(dates)):
        delta = (dates[i] - dates[i - 1]).days
        if delta < -2 or delta > 1:
            LOGGER.error(
                f"Date on line {i+1} ({dates[i]}) differs from previous date ({dates[i-1]}) by {delta} days, which seems unusual."
            )
            raise ValueError(
                "Dates in logfile are not in sensible order, please fix (expectation: Days difference between consecutive dates should be between -2 and 1), so from top to bottom you are expected to go from latest to earliest."
            )
    return len(dates)


def parse_date_line(line: str, line_number: Optional[int] = None) -> date:
    """
    Parses a date line and returns a date object.
    """
    assert line.startswith("# date: ")
    date_str = line.strip()[7:].strip()
    try:
        parsed_date = parse_isoformat(date_str + "T00:00:00").date()
        return parsed_date
    except Exception as e:
        if line_number is not None:
            LOGGER.error(
                f"Could not parse date from line {line_number}:\n\t[{e}]\n\t{line.strip()}"
            )
        raise e


def _filter_log_lines(lines: List[str], num_dates: int) -> Dict[int, str]:
    """
    Filters log lines to only include those starting with 'vw' or '# date: '.

    Parameters
    ----------
    lines : List[str]
        List of lines from the logfile.

    Returns
    -------
    Dict[int, str]
        A dictionary of original line numbers to filtered lines.
    """
    proper_lines = {
        i: l
        for i, l in enumerate(lines, start=1)
        if l.startswith("vw") or l.startswith("# date: ")
    }
    if len(proper_lines) - num_dates <= 0:
        LOGGER.error(
            "No valid log lines (expected format: Line starting with 'vw') found after filtering."
        )
        raise ValueError(
            "No valid log lines found after filtering. Expecting lines to start with 'vw'."
        )
    LOGGER.info(f"Filtered log lines: {len(proper_lines)} out of {len(lines)} kept.")
    return proper_lines


def filter_and_clean_logfile(
    logfile_path: Path, outpath: Optional[Path] = None
) -> Dict[int, str]:
    """
    Sanitizes a logfile by removing lines
    that do not start with 'vw' or '# date: ', and checking whether the date
    lines are correctly formatted and in a sensible order.

    Parameters
    ----------
    logfile_path : Path
        Path to the original logfile.
    outpath : Optional[Path]
        Path to save the sanitized logfile. If None, saves to the same directory with '_sanitized' suffix.

    Returns
    -------
    Dict[int, str]
        A dictionary of original line numbers to sanitized lines.
    """
    logfile_path = Path(logfile_path)
    assert logfile_path.exists(), f"No log file found at {logfile_path}."
    if outpath is None:
        outpath = (
            logfile_path.parent / f"{logfile_path.stem}_sanitized{logfile_path.suffix}"
        )
    else:
        outpath = Path(outpath)
        assert (
            outpath.parent.exists()
        ), f"Output directory {outpath.parent} does not exist."
        assert (
            outpath.suffix == logfile_path.suffix
        ), "Output file must have the same suffix as the input logfile."

    with logfile_path.open("r", encoding="utf-8", errors="ignore") as infile:
        lines = infile.readlines()
    num_dates = _check_date_order(lines)
    proper_lines = _filter_log_lines(lines, num_dates)
    with outpath.open("w", encoding="utf-8") as outfile:
        outfile.writelines(proper_lines.values())
    msg = (
        f"Please check whether sanitized logfile saved to {outpath} looks as expected."
    )
    LOGGER.info(msg)
    return proper_lines
