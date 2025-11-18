from dataclasses import dataclass, field
from math import isnan
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from ..calculations import get_clipped_mask, get_clipped_mask_by_distance
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
        if any(isnan(c) for c in self.observation.fiducial_coords):
            raise ValueError("Observation has no valid fiducial coordinates.")
        self.frames = self.observation.timeslot.load_guider_frames()
        self._fit_all()

    def __len__(self) -> int:
        return len(self.frames)

    def _fit_all(
        self,
        use_prev_as_guess: bool = False,
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
    def guider_times(self) -> np.ndarray:
        return np.array([f.ut_time for f in self.frames])

    @staticmethod
    def get_combined_stats_df(sequences: List["GuiderSequence"]) -> pd.DataFrame:
        """Converts a list of GuiderSequences to a pandas DataFrame."""
        centroids = np.array([s.get_centroid_stats() for s in sequences])
        cent_means = centroids[:, 0, :]
        cent_stds = centroids[:, 1, :]
        fwhms = np.array([s.get_fwhm_stats() for s in sequences])
        fwhm_means = fwhms[:, 0]
        fwhm_stds = fwhms[:, 1]

        data = {
            "filename": [s.observation.filename for s in sequences],
            "centroid_x_mean": cent_means[:, 0],
            "centroid_y_mean": cent_means[:, 1],
            "fwhm_mean": fwhm_means,
            "centroid_x_std": cent_stds[:, 0],
            "centroid_y_std": cent_stds[:, 1],
            "fwhm_std": fwhm_stds,
        }
        return pd.DataFrame(data)

    def get_fwhms_arcsec(self, sigmaclip_val: Optional[float] = 2.5) -> np.ndarray:
        if sigmaclip_val is None:
            return np.array([m.fwhm_arcsec for m in self.models])
        fwhms = self.get_fwhms_arcsec(sigmaclip_val=None)
        return fwhms[get_clipped_mask(fwhms, sigmaclip_val=sigmaclip_val)]

    def get_centroids(self, sigmaclip_val: Optional[float] = 2.5) -> np.ndarray:
        """Returns an array of (x, y) centroids from the fitted models."""
        if sigmaclip_val is None:
            return np.array([np.array((m.x_cent, m.y_cent)) for m in self.models])
        centroids = self.get_centroids(sigmaclip_val=None)
        return centroids[
            get_clipped_mask_by_distance(centroids, sigmaclip_val=sigmaclip_val)
        ]

    def get_centroid_stats(
        self, sigmaclip_val: Optional[float] = 2.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the mean and stddev (x, y) centroid from the fitted models, sigma-clipped if desired."""
        centroids = self.get_centroids(sigmaclip_val=sigmaclip_val)
        if len(centroids) == 0:
            return np.array([np.nan, np.nan]), np.array([np.nan, np.nan])
        return np.mean(centroids, axis=0), np.std(centroids, axis=0)

    def get_fwhm_stats(
        self, sigmaclip_val: Optional[float] = 2.5
    ) -> Tuple[float, float]:
        """Returns the mean and std FWHM val (in arcsec) from the fitted models, sigma-clipped if desired."""
        fwhms = self.get_fwhms_arcsec(sigmaclip_val=sigmaclip_val)
        if len(fwhms) == 0:
            return np.nan, np.nan
        return float(np.mean(fwhms)), float(np.std(fwhms))

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

    def plot_centroid_positions(
        self,
        relative_to: Literal["origin", "fiducial", "mean"] = "fiducial",
        annotate_mean: bool = True,
        **scatter_kwargs,
    ):
        """Scatter plot of fitted centroids from the sequence.

        Further documentation in `scatter_positions`.
        """
        from ..plotting import plot_centroid_positions

        return plot_centroid_positions(
            self, relative_to, annotate_mean=annotate_mean, **scatter_kwargs
        )

    def plot_fwhm_timeseries(self, annotate_mean: bool = True, **scatter_kwargs):
        """Plots the FWHM (in arcsec) as a function of time."""
        from ..plotting import plot_fwhm_sequence

        return plot_fwhm_sequence(self, **scatter_kwargs)
