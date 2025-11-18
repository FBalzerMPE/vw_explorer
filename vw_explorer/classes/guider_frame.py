from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from astropy.io import fits

from ..constants import DATA_PATH, GUIDER_DIR
from ..io.guider_indexing import create_guider_index, load_guider_index
from .star_model_fit import GuideStarModel


@dataclass
class GuiderFrame:
    """Loads and a guider frame with its associated data."""

    frame_path: Path
    ut_time: datetime = field(init=False)
    data: np.ndarray = field(repr=False, init=False)
    header_data: dict = field(repr=False, init=False)

    def __post_init__(self):
        self.frame_path = Path(self.frame_path)
        if not self.frame_path.exists():
            self.frame_path = DATA_PATH / self.frame_path
        if not self.frame_path.exists():
            raise FileNotFoundError(f"Guider frame not found: {self.frame_path}")
        self.data = fits.getdata(self.frame_path)  # type: ignore
        self.header_data = fits.getheader(self.frame_path)  # type: ignore
        date = self.header_data["DATE-OBS"]
        time = self.header_data["UT"]
        dt = date + "T" + time
        self.ut_time = datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S.%f")

    @staticmethod
    def get_guider_index(guider_dir: Path = GUIDER_DIR, **kwargs) -> pd.DataFrame:
        """Creates and loads the guider index."""
        if "silent" not in kwargs:
            kwargs["silent"] = True
        create_guider_index(guider_dir, **kwargs)
        return load_guider_index(guider_dir)

    def get_cutout_coords(
        self, center_x: float, center_y: float, size: float
    ) -> Tuple[int, int, int, int]:
        """Returns the coordinates of a square cutout from the frame data."""
        half_size = size / 2
        xmin = int(center_x - half_size)
        xmax = int(center_x + half_size)
        ymin = int(center_y - half_size)
        ymax = int(center_y + half_size)
        return xmin, xmax, ymin, ymax

    def get_cutout(self, center_x: float, center_y: float, size: float) -> np.ndarray:
        """Extracts a square cutout from the frame data."""
        xmin, xmax, ymin, ymax = self.get_cutout_coords(center_x, center_y, size)
        return self.data[ymin:ymax, xmin:xmax].copy()

    def get_model_fit(
        self, x_cent_in: float, y_cent_in: float, size: float = 70
    ) -> "GuideStarModel":
        """Fits a 2D Gaussian to a cutout around the specified center."""
        cutout = self.get_cutout(x_cent_in, y_cent_in, size)
        x_cent_in += 1  # FITS to numpy index correction
        y_cent_in += 1  # FITS to numpy index correction
        return GuideStarModel(
            inupt_data=cutout,
            x_cent_in=x_cent_in,
            y_cent_in=y_cent_in,
            size_in=size,
        )
