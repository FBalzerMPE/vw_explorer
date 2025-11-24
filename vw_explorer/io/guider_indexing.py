# Python
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from astropy.io import fits

from ..constants import CONFIG
from ..logger import LOGGER
from ..util import parse_isoformat


def create_guider_index(
    output_csv: Optional[Path] = None,
    remove_nonexistent: bool = False,
    force_reload: bool = False,
    silent: bool = False,
):
    """Creates a CSV index of guider FITS files with their observation date and time.
    The function scans the specified directory and its immediate subdirectories for FITS files,
    extracts the observation date and time from their headers, and writes this information to a CSV file.
    """
    g_fpath = CONFIG.guider_dir
    assert g_fpath.exists(), f"Guider directory {g_fpath} does not exist."
    if output_csv is None:
        output_csv = g_fpath / "guider_index.csv"
    if output_csv.is_dir():
        output_csv = output_csv / "guider_index.csv"
    assert output_csv.suffix == ".csv", "Output file must have a .csv extension."
    old_index_df: Optional[pd.DataFrame] = None
    if output_csv.exists() and not force_reload:
        old_index_df = pd.read_csv(output_csv)
        if not silent:
            LOGGER.info(
                f"Reading existing guider index with {len(old_index_df)} rows from {output_csv}"
            )
        if remove_nonexistent:
            # Remove entries for files that no longer exist
            existing_files = []
            for fpath in old_index_df["fname"].tolist():
                if Path(fpath).exists():
                    existing_files.append(fpath)
            if len(existing_files) < len(old_index_df):
                if not silent:
                    LOGGER.info(
                        f"Removing {len(old_index_df) - len(existing_files)} entries for non-existent files."
                    )
                old_index_df = old_index_df[
                    old_index_df["fname"].isin(existing_files)
                ].reset_index(drop=True)
                old_index_df.to_csv(output_csv, index=False)

    files = sorted(g_fpath.glob("*.fits"), key=lambda x: x.stat().st_mtime)
    for dirpath in [d for d in g_fpath.iterdir() if d.is_dir()]:
        files.extend(sorted(dirpath.glob("*.fits"), key=lambda x: x.stat().st_mtime))
    if old_index_df is not None:
        prev_files = set(old_index_df["fname"].tolist())
        files = [f for f in files if str(f) not in prev_files]
    if len(files) == 0:
        if not silent:
            LOGGER.info("No new files to index for guider index, skipping operation.")
        return
    LOGGER.info(
        f"Guider index creation: Found {len(files)} new files to index in {g_fpath}"
    )
    if len(files) > 500:
        LOGGER.warning(
            f"Large number of files ({len(files)}) to index. Creating index may take a while."
        )

    rows = []
    for f in files:
        hdr = fits.getheader(f, ignore_missing_end=True)
        try:
            date = hdr["DATE-OBS"]
            time = hdr["UT"]
            dt = parse_isoformat(f"{date}T{time}")
            date = dt.date().isoformat()
            time = dt.time().isoformat()
        except Exception as e:
            # fallback: use file modification time
            mdt = datetime.fromtimestamp(f.stat().st_mtime)
            if date is None:
                date = mdt.date().isoformat()
            if time is None:
                time = mdt.time().isoformat()
            LOGGER.warning(
                f"Could not read DATE-OBS/UT from header of {f}. Using file modification time. {e}"
            )
        rows.append({"date": date, "time": time, "fname": str(f)})
    df = pd.DataFrame(rows, columns=["date", "time", "fname"])
    if old_index_df is not None:
        df = pd.concat([old_index_df, df], ignore_index=True)
    df.to_csv(output_csv, index=False)
    if not silent:
        LOGGER.info(f"Wrote {len(df)} rows to {output_csv}")


def load_guider_index(index_csv: Path = CONFIG.guider_dir) -> pd.DataFrame:
    """Loads the guider index CSV file into a pandas DataFrame.
    You may provide either the CSV file path or the directory containing it.
    """
    assert index_csv.exists(), f"Index file {index_csv} does not exist."
    if index_csv.is_dir():
        index_csv = index_csv / "guider_index.csv"
        assert index_csv.exists(), f"Index file {index_csv} does not exist."
    df = pd.read_csv(index_csv)
    df["time"] = df["time"].str.slice(0, 8)  # keep only HH:MM:SS
    df["datetime"] = pd.to_datetime(
        df["date"] + "T" + df["time"], format="%Y-%m-%dT%H:%M:%S"
    )
    return df.sort_values("datetime").reset_index(drop=True)
