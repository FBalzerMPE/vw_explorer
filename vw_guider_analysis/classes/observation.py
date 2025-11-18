import math
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from .obs_timeslot import ObsTimeslot


def _sanitize_fpaths(f_in: str) -> list[str]:
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


def _parse_float(s: str) -> float:
    if s == "" or s == "-" or s.lower() == "auto":
        return float("nan")
    try:
        return float(s.split("x")[0])  # Handle cases like '120x6'
    except ValueError as e:
        raise AssertionError(f"Could not convert '{s}' to float.") from e


_EXPECTED_COLS = {
    "files": _sanitize_fpaths,
    "UT": _sanitize_start_time,
    "target": str,
    "exptime": _parse_float,
    "focus": _parse_float,
    "FWHM": _parse_float,
    "fiducial": _parse_fiducial_coords,
    "AM": _parse_float,
    "comments": str,
}


@dataclass
class Observation:
    files: List[Path]
    """List of file paths associated with the observation."""
    start_time_ut: datetime
    """Start time of the observation in UT."""
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
    airmass: float
    """Airmass noted for the observation."""
    comments: str
    """Additional comments or notes about the observation."""
    timeslot: Optional[ObsTimeslot] = field(init=False)
    """Timeslot during which the observation took place."""

    def __post_init__(self):
        if math.isnan(self.exptime):
            self.timeslot = None
        else:
            overhead = 90 * (len(self.files) - 1)  # 90 seconds overhead per frame
            full_exptime = self.exptime * len(self.files) + overhead
            self.timeslot = ObsTimeslot.from_start_and_time(
                self.start_time_ut, full_exptime
            )

    @classmethod
    def from_line(
        cls, line: str, date: date, base_datapath: Optional[Path] = None
    ) -> "Observation":
        num_max_parts = len(_EXPECTED_COLS) - 1  # comments can have spaces
        parts = line.strip().split()
        if len(parts) < num_max_parts:
            raise AssertionError("Log line does not contain all expected columns.")
        entry = {}
        for i, (key, sanitizer) in enumerate(_EXPECTED_COLS.items()):
            if key == "comments":
                continue
            entry[key] = sanitizer(parts[i])
        entry["comments"] = " ".join(parts[num_max_parts:])
        if entry["comments"] == "-":
            entry["comments"] = ""
        fpaths = (
            [base_datapath / (f + ".fits") for f in entry["files"]]
            if base_datapath
            else [Path(f + ".fits") for f in entry["files"]]
        )
        return cls(
            files=fpaths,
            start_time_ut=datetime.combine(date, entry["UT"]),
            target=entry["target"],
            exptime=entry["exptime"],
            focus=entry["focus"],
            fwhm=entry["FWHM"],
            fiducial_coords=entry["fiducial"],
            airmass=entry["AM"],
            comments=entry["comments"],
        )

    @classmethod
    def from_series(
        cls, series: pd.Series, base_datapath: Optional[Path] = None
    ) -> "Observation":
        """Creates an Observation from a pandas Series."""
        # Parse files, which are usually saved as string representations of lists
        files = series["files"]
        if isinstance(files, str):
            files = files.strip("[]").replace("'", "").split(", ")
        fpaths = (
            [base_datapath / f for f in files]
            if base_datapath
            else [Path(f) for f in files]
        )
        time = datetime.fromisoformat(series["start_time_ut"])
        return cls(
            files=fpaths,
            start_time_ut=time,
            target=series["target"],
            exptime=series["exptime"],
            focus=series["focus"],
            fwhm=series["fwhm"],
            fiducial_coords=(series["fiducial_x"], series["fiducial_y"]),
            airmass=series["airmass"],
            comments=series["comments"],
        )

    @staticmethod
    def to_dataframe(observations: List["Observation"]) -> pd.DataFrame:
        """Converts a list of Observations to a pandas DataFrame."""
        data = []
        for obs in observations:
            data.append(
                {
                    "uid": obs.uid,
                    "files": [f.name for f in obs.files],
                    "num_frames": obs.num_frames,
                    "target": obs.target,
                    "start_time_ut": obs.start_time_ut,
                    "exptime": obs.exptime,
                    "focus": obs.focus,
                    "fwhm": obs.fwhm,
                    "fiducial_x": obs.fiducial_coords[0],
                    "fiducial_y": obs.fiducial_coords[1],
                    "airmass": obs.airmass,
                    "comments": obs.comments,
                }
            )
        return pd.DataFrame(data)

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame, base_datapath: Optional[Path] = None
    ) -> List["Observation"]:
        """Creates a list of Observations from a pandas DataFrame."""
        observations = []
        for _, row in df.iterrows():
            observations.append(
                Observation.from_series(row, base_datapath=base_datapath)
            )
        return observations

    @property
    def uid(self) -> str:
        if self.num_frames > 1:
            return f"{self.files[0].stem}_{self.files[-1].stem}"
        return self.files[0].stem

    @property
    def long_name(self) -> str:
        s = f"{self.target} ({self.uid}, {self.start_time_ut.strftime('%Y-%m-%d %H:%M:%S UT')})"
        return s

    @property
    def num_frames(self) -> int:
        return len(self.files)

    @property
    def is_sky_obs(self) -> bool:
        return not math.isnan(self.airmass)

    @property
    def summary(self) -> str:
        if self.num_frames > 1:
            s = f"Observation: {self.uid} - {self.files[-1].stem} ({self.num_frames} frames)\n"
        else:
            s = f"Observation: {self.uid}\n"
        s += f"  Target: {self.target}\n"
        if self.timeslot:
            s += f"  Time slot: {self.timeslot.summary}\n"
        else:
            s += f"  Start time (UT): {self.start_time_ut.isoformat()}\n"
        if self.is_sky_obs:
            s += f"  Airmass: {self.airmass:.2f}\n"
            s += f"  FWHM: {self.fwhm:.2f} arcsec\n"
        s += f"  Number of frames: {self.num_frames}\n"
        if not math.isnan(self.exptime):
            s += f"  Exposure time per frame: {self.exptime:.1f} s\n"
        if self.comments:
            s += f"  Comments: {self.comments}\n"
        s += f"  Fiducial coords: ({self.fiducial_coords[0]:.1f}, {self.fiducial_coords[1]:.1f})\n"
        return s
