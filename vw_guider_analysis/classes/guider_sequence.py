from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np
import pandas as pd

from ..calculations import get_mean_centroid, get_mean_value
from .guider_frame import GuiderFrame
from .observation import Observation
from .star_model_fit import GuideStarModel


@dataclass
class GuiderSequence:
    """Represents a sequence of guider frames for analysis."""

    observation: Observation
    frames: List[GuiderFrame] = field(init=False, repr=False)
    models: List[GuideStarModel] = field(init=False, repr=False)

    def __post_init__(self):
        if self.observation.timeslot is None:
            raise ValueError("Observation has no valid timeslot for guider frames.")
        self.frames = self.observation.timeslot.load_guider_frames()
        self._fit_all()

    def __len__(self) -> int:
        return len(self.frames)

    def _fit_all(
        self,
        use_prev_as_guess: bool = True,
    ):
        """Fits all guider frames in the sequence."""
        self.models = []
        x_guess, y_guess = self.observation.fiducial_coords
        for gf in self.frames:
            m = gf.get_model_fit(x_guess, y_guess)
            self.models.append(m)
            if not use_prev_as_guess and not m.has_failed:
                continue
            x_guess = m.x_cent
            y_guess = m.y_cent

    @property
    def fwhms_arcsec(self) -> np.ndarray:
        return np.array([m.fwhm_arcsec for m in self.models])

    @property
    def guider_times(self) -> np.ndarray:
        return np.array([f.ut_time for f in self.frames])

    @staticmethod
    def to_dataframe(sequences: List["GuiderSequence"]) -> pd.DataFrame:
        """Converts a list of GuiderSequences to a pandas DataFrame."""
        centroids = np.array([s.get_mean_centroid() for s in sequences])
        fwhms = [s.get_mean_fwhm() for s in sequences]
        data = {
            "uid": [s.observation.uid for s in sequences],
            "mean_centroid_x": centroids[:, 0],
            "mean_centroid_y": centroids[:, 1],
            "mean_fwhm": fwhms,
        }
        return pd.DataFrame(data)

    def get_centroids(self) -> np.ndarray:
        """Returns an array of (x, y) centroids from the fitted models."""
        return np.array([np.array((m.x_cent, m.y_cent)) for m in self.models])

    def get_mean_centroid(self, sigmaclip_val: Optional[float] = 3) -> np.ndarray:
        """Returns the mean (x, y) centroid from the fitted models, sigma-clipped if desired."""
        return get_mean_centroid(self.get_centroids(), sigmaclip_val=sigmaclip_val)

    def get_mean_fwhm(self, sigmaclip_val: Optional[float] = 3) -> float:
        """Returns the mean FWHM (in arcsec) from the fitted models, sigma-clipped if desired."""
        return get_mean_value(self.fwhms_arcsec, sigmaclip_val=sigmaclip_val)

    def plot_fits(self, idx: Optional[int] = None):
        """Plots the guider frames with their fitted models.
        If idx is provided, only plots that specific frame.
        """
        from ..plotting import plot_guidefit_model

        if idx is not None:
            plot_guidefit_model(self.frames[idx], self.models[idx])
            return
        for frame, model in zip(self.frames, self.models):
            plot_guidefit_model(frame, model)

    def plot_positions(
        self,
        relative_to: Literal["origin", "fiducial", "mean"] = "fiducial",
        annotate_mean: bool = True,
        **scatter_kwargs,
    ):
        """Scatter plot of fitted centroids from the sequence.

        Further documentation in `scatter_positions`.
        """
        from ..plotting import scatter_positions

        x_0: Optional[float] = None
        y_0: Optional[float] = None
        if relative_to == "fiducial":
            x_0, y_0 = self.observation.fiducial_coords
        elif relative_to == "mean":
            x_0, y_0 = self.get_mean_centroid()
        ax = scatter_positions(self, x_0=x_0, y_0=y_0, **scatter_kwargs)
        if annotate_mean:
            mean_x, mean_y = self.get_mean_centroid()
            if relative_to in ["fiducial", "mean"]:
                mean_x -= x_0
                mean_y -= y_0
            ax.plot(mean_x, mean_y, "ro", markersize=10, alpha=0.5)

    def plot_fwhm_timeseries(self, annotate_mean: bool = True, **scatter_kwargs):
        """Plots the FWHM (in arcsec) as a function of time."""
        from ..plotting import plot_fwhm_sequence

        ax = plot_fwhm_sequence(self, **scatter_kwargs)
        if not annotate_mean:
            return
        mean_fwhm = self.get_mean_fwhm()
        ax.axhline(
            mean_fwhm,
            color="red",
            linestyle="--",
        )
        ax.text(
            0.95,
            0.95,
            f"Mean FWHM: {mean_fwhm:.2f} arcsec",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color="red",
        )
