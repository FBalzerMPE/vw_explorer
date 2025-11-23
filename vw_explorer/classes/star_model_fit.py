from dataclasses import dataclass, field

import numpy as np
from astropy.modeling import Model

from ..calculations import fit_guide_star
from ..constants import GUIDER_PIXSCALE


@dataclass
class GuideStarModel:
    input_data: np.ndarray = field(repr=False)
    """The cutout data used for fitting."""
    x_cent_in: float
    """The centroid x position used for the cutout (input coords)."""
    y_cent_in: float
    """The centroid y position used for the cutout (input coords)."""
    size_in: float
    """The size of the cutout used for fitting."""
    exptime: float
    """The exposure time of the frame."""
    model: Model = field(repr=False, init=False)
    """The fitted model."""

    def __post_init__(self):
        self.model = fit_guide_star(
            self.input_data,
            stddev_guess=3.0,
            window=20,
        )

    @property
    def has_failed(self) -> bool:
        """Indicates whether the fit has failed."""
        return self.fwhm_pix > 30 or np.isnan(self.x_cent) or np.isnan(self.y_cent)

    @property
    def x_cent(self) -> float:
        """The fitted x center in the original frame coordinates."""
        return self.model.x_mean_0 + self.x_cent_in - self.size_in / 2  # type: ignore

    @property
    def y_cent(self) -> float:
        """The fitted y center in the original frame coordinates."""
        return self.model.y_mean_0 + self.y_cent_in - self.size_in / 2  # type: ignore

    @property
    def amplitude(self) -> float:
        """The fitted amplitude of the Gaussian, divided by exposure time."""
        return self.model.amplitude_0  # type: ignore

    @property
    def total_flux_rate(self) -> float:
        """The total flux of the fitted Gaussian, divided by exposure time.

        The total flux for a 2D Gaussian is the integral of this function over all x and y, which evaluates to:

            Flux = A * 2 * pi * sigma_x * sigma_y

        Where:
            A        : Amplitude (peak value).
            sigma_x  : Standard deviation in the x-direction.
            sigma_y  : Standard deviation in the y-direction.
        For the flux rate, we divide by the exposure time.
        """
        x_std = self.model.x_stddev_0  # type: ignore
        y_std = self.model.y_stddev_0  # type: ignore
        flux = 2 * np.pi * x_std * y_std * self.model.amplitude_0  # type: ignore
        return flux / self.exptime

    @property
    def fwhm_pix(self) -> float:
        """The fitted full-width at half-maximum of the Gaussian."""
        # return 2.355 * self.model.stddev_0  # type: ignore
        x_std = self.model.x_stddev_0  # type: ignore
        y_std = self.model.y_stddev_0  # type: ignore
        # comb_std = (x_std + y_std) / 2.0
        comb_std = np.sqrt(x_std * y_std)
        comb_std = min(x_std, y_std)
        return 2.355 * comb_std

    @property
    def fwhm_arcsec(self) -> float:
        """The fitted FWHM in arcseconds."""
        return self.fwhm_pix * GUIDER_PIXSCALE

    def get_residuals(self) -> np.ndarray:
        """Calculates the residuals between the input data and the fitted model."""
        y, x = np.mgrid[0 : self.input_data.shape[0], 0 : self.input_data.shape[1]]
        fitted_data = self.model(x, y)
        resid = self.input_data - fitted_data

        # preserve NaNs from input
        resid[~np.isfinite(self.input_data)] = np.nan
        return resid
