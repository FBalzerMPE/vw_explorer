import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from astropy.io import fits

from ..constants import CALIB_NAMES, OBS_PATH
from ..io import parse_vw_filenames
from ..logger import LOGGER
from ..util import parse_isoformat
from .obs_timeslot import ObsTimeslot


def _sanitize_fnames(f_in: str) -> List[str]:
    assert f_in.startswith("vw"), f"Filename {f_in} does not start with 'vw'."
    return parse_vw_filenames(f_in)


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


def _try_find_file(fname: str, base_datapath: Optional[Path] = None) -> Path:
    if base_datapath is None:
        return Path(fname).with_suffix(".fits")
    # Walk the base datapath to find the file
    avail = list(base_datapath.glob("**/" + fname + ".fits"))
    if avail:
        assert len(avail) == 1, f"Multiple files found for {fname} in {base_datapath}."
        return avail[0]
    return (base_datapath / fname).with_suffix(".fits")


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
    fpath: Path = field(repr=False)
    """Path to the observation file."""
    target: str
    """Target of the observation."""
    start_time_ut: datetime
    """UT start time of the observation as noted in the observer log."""
    exptime: float
    """Exposure time of the observation in seconds."""
    focus: float
    """Focus value of the observation."""
    fwhm_noted: float
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
            else ObsTimeslot.from_start_and_time(self.start_time_ut, self.exptime)
        )

    @classmethod
    def from_fits(
        cls, fpath: Path, fid_x: float = float("nan"), fid_y: float = float("nan")
    ) -> "Observation":
        """Creates an Observation instance from a FITS file."""
        fpath = Path(fpath)
        assert fpath.exists(), f"FITS file not found: {fpath}"
        header = fits.getheader(fpath)  # type: ignore
        filename = fpath.stem
        start_time_str = header.get("DATE-OBS", "")
        start_time = datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M:%S.%f")
        t_parts = header.get("OBJECT", "Unknown").split("dither")
        target = t_parts[0].strip()
        dither = int(t_parts[1].strip()) + 1 if len(t_parts) > 1 else 1
        exptime = header.get("EXPTIME", float("nan"))
        focus = header.get("FOCUS", float("nan"))
        airmass = header.get("AIRMASS", float("nan"))

        return cls(
            filename=filename,
            fpath=fpath,
            start_time_ut=start_time,
            target=target,
            exptime=exptime,
            focus=focus,
            fwhm_noted=float("nan"),
            fiducial_coords=(fid_x, fid_y),
            airmass_noted=airmass,
            comments=header.get("COMMENT", ""),
            dither=dither,
        )

    @staticmethod
    def parse_obs_log_line(line: str, date: date) -> List["Observation"]:
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
            fpath = _try_find_file(fname, OBS_PATH)
            actual_dither = dither + i
            start_time = datetime.combine(date, entry["UT"])
            if not math.isnan(exptime):
                # Adding the 90 seconds overhead only yields approximate start time!
                start_time += timedelta(seconds=i * (exptime + 90))
            obs = Observation(
                filename=fname,
                fpath=fpath,
                start_time_ut=start_time,
                target=target,
                exptime=exptime,
                focus=entry["focus"],
                fwhm_noted=entry["FWHM"],
                fiducial_coords=entry["fiducial"],
                airmass_noted=entry["AM"],
                comments=entry["comments"],
                dither=actual_dither,
            )
            obs._update_information(silent=True)
            observations.append(obs)
        return observations

    @classmethod
    def from_series(cls, series: pd.Series) -> "Observation":
        """Creates an Observation from a pandas Series."""
        # Parse files, which are usually saved as string representations of lists
        fname = series["filename"]
        fpath = Path(series["fpath"])
        time = parse_isoformat(series["start_time_ut"])
        return cls(
            filename=fname,
            fpath=fpath,
            start_time_ut=time,
            target=series["target"],
            exptime=series["exptime"],
            focus=series["focus"],
            fwhm_noted=series["fwhm_noted"],
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
                    "fpath_available": obs.file_available,
                    "dither": obs.dither,
                    "target": obs.target,
                    "start_time_ut": obs.start_time_ut,
                    "exptime": obs.exptime,
                    "focus": obs.focus,
                    "fwhm_noted": obs.fwhm_noted,
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
        s = f"{self.target} ({self.filename}, {self.start_time_ut.strftime('%Y-%m-%d %H:%M:%S UT')}, dither {self.dither})"
        return s

    @property
    def is_sky_obs(self) -> bool:
        return not math.isnan(self.airmass_noted)

    @property
    def is_calibration_obs(self) -> bool:
        """Is this observation a calibration frame (bias, dark, flat, etc.)?"""
        return any(name in self.target.lower() for name in CALIB_NAMES)

    @property
    def file_available(self) -> bool:
        """Checks if the observation file exists."""
        return self.fpath.is_file()

    @property
    def summary(self) -> str:
        s = f"Observation: {self.filename}\n"
        s += f"  Target: {self.target}, dither {self.dither}\n"
        if self.timeslot:
            s += f"  Time slot: {self.timeslot.summary}\n"
        else:
            s += f"  Start time (UT): {self.start_time_ut.isoformat()}\n"
        if self.is_sky_obs:
            s += f"  Airmass: {self.airmass_noted:.2f}\n"
            s += f"  FWHM: {self.fwhm_noted:.2f} arcsec\n"
        if not math.isnan(self.exptime):
            s += f"  Exposure time per frame: {self.exptime:.1f} s\n"
        if self.comments:
            s += f"  Comments: {self.comments}\n"
        s += f"  Fiducial coords: ({self.fiducial_coords[0]:.1f}, {self.fiducial_coords[1]:.1f})\n"
        return s

    @property
    def trimmed_comments(self) -> str:
        """Returns the comments trimmed to a maximum of 60 characters."""
        max_length = 60
        if len(self.comments) <= max_length:
            return self.comments
        return self.comments[: max_length - 3] + "..."

    def _update_information(self, silent=True) -> None:
        """Update the information of the observation from its FITS header."""
        if not self.fpath.is_file():
            if not silent:
                LOGGER.warning(
                    f"Cannot update observation info, file not found: {self.fpath}"
                )
            return
        header = fits.getheader(self.fpath)
        self.start_time_ut = datetime.strptime(
            header.get("DATE-OBS", ""), "%Y-%m-%dT%H:%M:%S.%f"
        )
        t_parts = header.get("OBJECT", "Unknown").split("dither")
        self.target = t_parts[0].strip()
        if len(t_parts) > 1:
            try:
                self.dither = int(t_parts[1].strip()) + 1
            except ValueError:
                self.dither = 1
        else:
            self.dither = 1
        self.exptime = header.get("EXPTIME", float("nan"))
        self.focus = header.get("FOCUS", float("nan"))
        self.timeslot = (
            None
            if math.isnan(self.exptime)
            else ObsTimeslot.from_start_and_time(self.start_time_ut, self.exptime)
        )
