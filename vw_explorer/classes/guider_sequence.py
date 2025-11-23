from dataclasses import dataclass, field
from math import isnan
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from typing_extensions import Literal

from ..calculations import get_clipping_kept_mask, get_clipping_kept_mask_by_distance
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

    def __str__(self) -> str:
        return f"GuiderSequence for {self.observation}"

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
            # Clear data after fitting to save memory
            gf.clear_data()

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
        flux_rates = np.array(
            [s.get_flux_rate_stats(sigmaclip_val=None) for s in sequences]
        )
        flux_rate_means = flux_rates[:, 0]
        flux_rate_stds = flux_rates[:, 1]
        data = {
            "filename": [s.observation.filename for s in sequences],
            "centroid_x_mean": cent_means[:, 0],
            "centroid_y_mean": cent_means[:, 1],
            "flux_rate_mean": flux_rate_means,
            "fwhm_mean": fwhm_means,
            "centroid_x_std": cent_stds[:, 0],
            "centroid_y_std": cent_stds[:, 1],
            "fwhm_std": fwhm_stds,
            "flux_rate_std": flux_rate_stds,
        }
        return pd.DataFrame(data)

    def get_flux_rates(self, sigmaclip_val: Optional[float] = None) -> np.ndarray:
        if sigmaclip_val is None:
            return np.array([m.total_flux_rate for m in self.models])
        flux_rates = self.get_flux_rates(sigmaclip_val=None)
        return flux_rates[get_clipping_kept_mask(flux_rates, sigmaclip_val=sigmaclip_val)]

    def get_flux_rate_stats(
        self, sigmaclip_val: Optional[float] = None
    ) -> Tuple[float, float]:
        flux_rates = self.get_flux_rates(sigmaclip_val=sigmaclip_val)
        if len(flux_rates) == 0:
            return np.nan, np.nan
        return float(np.mean(flux_rates)), float(np.std(flux_rates))

    def get_fwhms_arcsec(self, sigmaclip_val: Optional[float] = 2.5) -> np.ndarray:
        if sigmaclip_val is None:
            return np.array([m.fwhm_arcsec for m in self.models])
        fwhms = self.get_fwhms_arcsec(sigmaclip_val=None)
        return fwhms[get_clipping_kept_mask(fwhms, sigmaclip_val=sigmaclip_val)]

    def get_centroids(self, sigmaclip_val: Optional[float] = 2.5) -> np.ndarray:
        """Returns an array of (x, y) centroids from the fitted models."""
        if sigmaclip_val is None:
            return np.array([np.array((m.x_cent, m.y_cent)) for m in self.models])
        centroids = self.get_centroids(sigmaclip_val=None)
        return centroids[
            get_clipping_kept_mask_by_distance(centroids, sigmaclip_val=sigmaclip_val)
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

    def get_stacked_frame(self) -> np.ndarray:
        """Returns a normalized stacked frame from all guider frames in the sequence."""
        from ..calculations.image_stacking import stack_frames

        return stack_frames(
            self.frames, self.get_centroids(sigmaclip_val=None), sigmaclip_val=2.5
        )

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

    def plot_initial_frame(
        self,
        center_around: Literal["none", "fiducial", "mean"] = "fiducial",
        cutout_size: int = 70,
        **kwargs,
    ):
        """Plots the first guider frame in the sequence with its fitted model."""
        from ..plotting import plot_frame_cutout

        mean_coords = None
        if center_around == "mean":
            mean_coords = self.get_centroid_stats(sigmaclip_val=2.5)[0]
        elif center_around == "fiducial":
            mean_coords = self.observation.fiducial_coords
        else:
            kwargs["fid_coords"] = self.observation.fiducial_coords
        plot_frame_cutout(self.frames[0], mean_coords, cutout_size, **kwargs)

    def plot_centroid_positions(
        self,
        relative_to: Literal["origin", "fiducial", "mean"] = "fiducial",
        annotate_mean: bool = True,
        **kwargs,
    ):
        """Scatter plot of fitted centroids from the sequence.

        Further documentation in `scatter_positions`.
        """
        from ..plotting import plot_centroids_for_single_gseq

        return plot_centroids_for_single_gseq(
            self, relative_to, annotate_mean=annotate_mean, **kwargs
        )

    def plot_fwhm_timeseries(self, annotate_mean: bool = True, **scatter_kwargs):
        """Plots the FWHM (in arcsec) as a function of time."""
        from ..plotting import plot_fwhms_for_single_gseq

        return plot_fwhms_for_single_gseq(self, **scatter_kwargs)

    def plot_flux_rate_timeseries(self, annotate_mean: bool = True, **scatter_kwargs):
        """Plots the flux rates as a function of time."""
        from ..plotting import plot_flux_rates_for_single_gseq

        return plot_flux_rates_for_single_gseq(
            self, annotate_mean=annotate_mean, **scatter_kwargs
        )

    def plot_summary(self):
        """Plots summary statistics for the guider sequence."""
        from ..plotting import plot_guider_sequence_summary

        return plot_guider_sequence_summary(self)
