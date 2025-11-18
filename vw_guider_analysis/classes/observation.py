import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from .obs_timeslot import ObsTimeslot


def _sanitize_fnames(f_in: str) -> list[str]:
    """Parse the given filename string, which may contain a range indicated by a hyphen.
    Returns a list of individual filenames.
    Examples:
        "vw004123-125" -> ["vw004123", "vw004124", "vw004125"]
        "vw004200" -> ["vw004200"]
        "vw001432-34" -> ["vw001432", "vw001433", "vw001434"]
    """
    assert f_in.startswith("vw"), f"Filename {f_in} does not start with 'vw'."
    if "-" not in f_in:
        return [f_in]
    main = f_in.split("-")[0]
    num_start = int(main[2:])
    num_end = int(f_in.split("-")[1])
    # Only relevant digits are replaced
    num_end_digits = len(str(num_end))
    num_start_part = str(num_start)[:-num_end_digits]
    num_end = int(f"{num_start_part}{num_end}")
    assert num_start < num_end, f"Invalid file range: {f_in}"
    return [f"vw{n:0>6}" for n in range(num_start, num_end + 1)]


def _sanitize_start_time(ut_str: str) -> time:
    # Basic validation for UT time format HH:MM:SS
    parts = ut_str.split(":")
    if len(parts) == 2:
        parts.append("0")  # Add seconds if missing
    if len(parts) != 3:
        raise AssertionError("UT time must be in HH:MM or HH:MM:SS format.")
    hour, minute, second = parts
    try:
        t = time(int(hour), int(minute), int(second))
        return t
    except ValueError as e:
        raise AssertionError(f"UT time of {ut_str} contains invalid values.") from e


def _parse_fiducial_coords(fid_str: str) -> Tuple[float, float]:
    if fid_str == "" or fid_str == "-":
        return (float("nan"), float("nan"))
    try:
        x_str, y_str = fid_str.split(",")
        x = float(x_str)
        y = float(y_str)
        return (x, y)
    except Exception as e:
        raise AssertionError(
            "Fiducial coordinates must be in 'x,y' format with numeric values."
        ) from e


def _parse_target_and_dither(target_str: str) -> Tuple[str, int]:
    """Parses the target string to extract the target name and dither position.
    Examples:
        "M52" -> ("M52", 1)
        "M52_D2" -> ("M52", 2)
        "M52_3" -> ("M52", 3)
    """
    if "_" not in target_str:
        return (target_str, 1)
    parts = target_str.rsplit("_", 1)
    target_name = parts[0]
    try:
        dither_pos = int(parts[1].lower().lstrip("d"))
        assert (
            dither_pos > 0
        ), f"Dither position must be a positive integer, not {dither_pos} (parsed from {target_str})."
        return (target_name, dither_pos)
    except ValueError as e:
        raise AssertionError(
            f"Could not parse dither position from target string '{target_str}'."
        ) from e


def _parse_float(s: str) -> float:
    if s == "" or s == "-" or s.lower() == "auto":
        return float("nan")
    try:
        return float(s.split("x")[0])  # Handle cases like '120x6'
    except ValueError as e:
        raise AssertionError(f"Could not convert '{s}' to float.") from e


_EXPECTED_COLS = {
    "files": _sanitize_fnames,
    "UT": _sanitize_start_time,
    "target_and_dither": _parse_target_and_dither,
    "exptime": _parse_float,
    "focus": _parse_float,
    "FWHM": _parse_float,
    "fiducial": _parse_fiducial_coords,
    "AM": _parse_float,
    "comments": str,
}


@dataclass
class Observation:
    filename: str
    """The stem of the file associated with the observation, in the format vw######. Use fpath to get full path.
    """
    fpath: Path
    """Path to the observation file."""
    start_time_ut_noted: datetime
    """UT start time of the observation as noted in the observer log."""
    target: str
    """Target of the observation."""
    exptime: float
    """Exposure time of the observation in seconds."""
    focus: float
    """Focus value of the observation."""
    fwhm: float
    """Full width at half maximum (FWHM) in arcseconds noted during observation."""
    fiducial_coords: Tuple[float, float]
    """Fiducial coordinates at guider in pixels (x, y)."""
    airmass_noted: float
    """Airmass noted for the observation."""
    comments: str
    """Additional comments or notes about the observation."""
    dither: int = 1
    """The dither position of the observation, if applicable.
    Indexing starts at 1."""
    timeslot: Optional[ObsTimeslot] = field(init=False)
    """Timeslot during which the observation took place."""

    def __post_init__(self):
        self.timeslot = (
            None
            if math.isnan(self.exptime)
            else ObsTimeslot.from_start_and_time(self.start_time_ut_noted, self.exptime)
        )

    @staticmethod
    def parse_obs_log_line(
        line: str, date: date, base_datapath: Optional[Path] = None
    ) -> List["Observation"]:
        """Parses a single line from an observation log and returns a list of Observations."""
        num_max_parts = len(_EXPECTED_COLS) - 1  # comments can have spaces
        parts = line.strip().split()
        if len(parts) < num_max_parts:
            raise AssertionError(
                f"Log line only contains {len(parts)} columns, expected at least {num_max_parts}."
            )
        entry = {}
        for i, (key, sanitizer) in enumerate(_EXPECTED_COLS.items()):
            if key == "comments":
                continue
            entry[key] = sanitizer(parts[i])
        entry["comments"] = " ".join(parts[num_max_parts:])
        if entry["comments"] == "-":
            entry["comments"] = ""
        target, dither = entry["target_and_dither"]
        exptime = entry["exptime"]
        observations = []
        for i, fname in enumerate(entry["files"]):
            fpath = (base_datapath / fname) if base_datapath else Path(fname)
            fpath = fpath.with_suffix(".fits")
            actual_dither = dither + i
            start_time = datetime.combine(date, entry["UT"])
            if not math.isnan(exptime):
                # Adding the 90 seconds overhead only yields approximate start time!
                start_time += timedelta(seconds=i * (exptime + 90))
            obs = Observation(
                filename=fname,
                fpath=fpath,
                start_time_ut_noted=start_time,
                target=target,
                exptime=exptime,
                focus=entry["focus"],
                fwhm=entry["FWHM"],
                fiducial_coords=entry["fiducial"],
                airmass_noted=entry["AM"],
                comments=entry["comments"],
                dither=actual_dither,
            )
            observations.append(obs)
        return observations

    @classmethod
    def from_series(cls, series: pd.Series) -> "Observation":
        """Creates an Observation from a pandas Series."""
        # Parse files, which are usually saved as string representations of lists
        fname = series["filename"]
        fpath = Path(series["fpath"])
        time = datetime.fromisoformat(series["start_time_ut_noted"])
        return cls(
            filename=fname,
            fpath=fpath,
            start_time_ut_noted=time,
            target=series["target"],
            exptime=series["exptime"],
            focus=series["focus"],
            fwhm=series["fwhm"],
            fiducial_coords=(series["fiducial_x"], series["fiducial_y"]),
            airmass_noted=series["airmass_noted"],
            comments=series["comments"],
            dither=series["dither"],
        )

    @staticmethod
    def to_dataframe(observations: List["Observation"]) -> pd.DataFrame:
        """Converts a list of Observations to a pandas DataFrame."""
        data = []
        for obs in observations:
            data.append(
                {
                    "filename": obs.filename,
                    "fpath": str(obs.fpath),
                    "dither": obs.dither,
                    "target": obs.target,
                    "start_time_ut_noted": obs.start_time_ut_noted,
                    "exptime": obs.exptime,
                    "focus": obs.focus,
                    "fwhm": obs.fwhm,
                    "fiducial_x": obs.fiducial_coords[0],
                    "fiducial_y": obs.fiducial_coords[1],
                    "airmass_noted": obs.airmass_noted,
                    "comments": obs.comments,
                }
            )
        return pd.DataFrame(data)

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> List["Observation"]:
        """Creates a list of Observations from a pandas DataFrame."""
        observations = []
        for _, row in df.iterrows():
            observations.append(Observation.from_series(row))
        return observations

    @property
    def long_name(self) -> str:
        s = f"{self.target} ({self.filename}, {self.start_time_ut_noted.strftime('%Y-%m-%d %H:%M:%S UT')}, dither {self.dither})"
        return s

    @property
    def is_sky_obs(self) -> bool:
        return not math.isnan(self.airmass_noted)

    @property
    def summary(self) -> str:
        s = f"Observation: {self.filename}\n"
        s += f"  Target: {self.target}, dither {self.dither}\n"
        if self.timeslot:
            s += f"  Time slot: {self.timeslot.summary}\n"
        else:
            s += f"  Start time (UT): {self.start_time_ut_noted.isoformat()}\n"
        if self.is_sky_obs:
            s += f"  Airmass: {self.airmass_noted:.2f}\n"
            s += f"  FWHM: {self.fwhm:.2f} arcsec\n"
        if not math.isnan(self.exptime):
            s += f"  Exposure time per frame: {self.exptime:.1f} s\n"
        if self.comments:
            s += f"  Comments: {self.comments}\n"
        s += f"  Fiducial coords: ({self.fiducial_coords[0]:.1f}, {self.fiducial_coords[1]:.1f})\n"
        return s
