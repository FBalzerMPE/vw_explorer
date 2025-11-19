from pathlib import Path
from typing import List, Optional

from ..constants import OBS_PATH
from ..logger import LOGGER


def parse_vw_filenames(f_in: str, add_fits_extension: bool = False) -> List[str]:
    """Parse the given filename string, which may contain a range indicated by a hyphen.
    The filename format is assumed to either be 'vwXXXXXX', where 'X' are digits, or a range
    like 'vwXXXXXX-YYY', where 'YYY' indicates the last digits of the range.
    The vw prefix and leading zeros are optional in the input.
    Returns a list of individual filenames.
    Examples:
        "vw004123-125" -> ["vw004123", "vw004124", "vw004125"]
        "vw004200" -> ["vw004200"]
        "vw001432-34" -> ["vw001432", "vw001433", "vw001434"]
    """
    f_in = f_in.strip().lower().replace(".fits", "").replace("vw", "")
    leading, fname = str(Path(f_in).parent), Path(f_in).name
    f_in = str(fname)
    if "-" not in f_in:
        try:
            f_in = f"vw{int(f_in):0>6}"
        except ValueError:
            raise AssertionError(f"Invalid filename: {f_in}")
        if add_fits_extension:
            f_in += ".fits"
        return [str(Path(leading) / f_in)] if leading else [f_in]
    assert f_in.count("-") == 1, f"Invalid filename range: {f_in}"
    main, extra = f_in.split("-")
    num_start = int(main)
    num_end = int(extra.strip())
    # Only relevant digits are replaced
    num_end_digits = len(str(num_end))
    num_start_part = str(num_start)[:-num_end_digits]
    num_end = int(f"{num_start_part}{num_end}")
    assert num_start < num_end, f"Invalid file range: {f_in}"
    fnames = [
        f"vw{n:0>6}" + (".fits" if add_fits_extension else "")
        for n in range(num_start, num_end + 1)
    ]
    if not leading:
        return fnames
    return [str(Path(leading) / f) for f in fnames]


def _find_vw_files(
    filenames: List[Path], remove_nonexisting: bool = True
) -> List[Path]:
    """Check each filename for existence, and if not, try to identify its path in the OBS_PATH directory and subdirectory."""
    all_avail_files = set(OBS_PATH.rglob("vw*.fits"))
    fname_dict = {f.name: f for f in all_avail_files}
    not_avail = []
    existing_files = []
    for fname in filenames:
        if fname.exists():
            existing_files.append(fname)
            continue
        if fname.name in fname_dict:
            existing_files.append(fname_dict[fname.name])
            continue
        not_avail.append(fname)
        if not remove_nonexisting:
            existing_files.append(fname)
    if not_avail:
        LOGGER.warning(
            f"The following files were not found: {[str(f) for f in not_avail]}"
        )
    return existing_files


def infer_vw_filenames(f_in: Optional[str]) -> List[Path]:
    if f_in is not None:
        filenames = [Path(f) for f in parse_vw_filenames(f_in, add_fits_extension=True)]
        return _find_vw_files(filenames)
    fits_files = sorted(Path(".").glob("vw*.fits"))
    if not fits_files:
        LOGGER.error(
            "No FITS files found matching 'vw*.fits'. Please provide an input file."
        )
        raise FileNotFoundError("No FITS files following 'vw*.fits' format found.")
    return fits_files[-1:]
