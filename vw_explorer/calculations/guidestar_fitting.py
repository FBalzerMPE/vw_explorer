from typing import Optional

import numpy as np
from astropy.modeling import Fittable2DModel, Model, Parameter, fitting, models


class SymmetricGaussian2D(Fittable2DModel):
    """
    2D Gaussian with a single (isotropic) stddev parameter.
    amplitude, x_mean, y_mean, stddev are fit parameters.
    """

    amplitude = Parameter()
    x_mean = Parameter()
    y_mean = Parameter()
    stddev = Parameter()

    @staticmethod
    def evaluate(x, y, amplitude, x_mean, y_mean, stddev):
        return amplitude * np.exp(
            -0.5 * (((x - x_mean) / stddev) ** 2 + ((y - y_mean) / stddev) ** 2)
        )


def fit_guide_star(
    data: np.ndarray,
    stddev_guess: float = 3.0,
    window: int = 15,
    x_guess: Optional[float] = None,
    y_guess: Optional[float] = None,
) -> Model:
    """Fits a 2D Gaussian + constant background to the input data.
    Parameters
    ----------
    data : np.ndarray
        2D array of the cutout data to fit.
    x_guess : float
        Initial guess for the x center of the Gaussian.
    y_guess : float
        Initial guess for the y center of the Gaussian.
    stddev_guess : float, optional
        Initial guess for the standard deviation of the Gaussian, by default 3.0.
    Returns
    -------
    fitted : astropy.modeling.Model
        The fitted model.
    (x_mean, y_mean) : tuple[float, float]
        The fitted center coordinates.
    fwhm : float
        The fitted full-width at half-maximum of the Gaussian.
    """
    if data.size == 0:
        raise ValueError("Empty cutout - check coordinates / cutout_size")

    ny, nx = data.shape

    # find peak or use provided guess (guesses expected in global coords)
    peak_row, peak_col = np.unravel_index(np.nanargmax(data), data.shape)
    if x_guess is not None and y_guess is not None:
        # convert to int pixel location for window center
        peak_col = int(round(x_guess))
        peak_row = int(round(y_guess))

    half = max(1, window // 2)
    x_min = max(0, peak_col - half)
    x_max = min(nx, peak_col + half + 1)
    y_min = max(0, peak_row - half)
    y_max = min(ny, peak_row + half + 1)

    sub = data[y_min:y_max, x_min:x_max]
    if sub.size == 0:
        raise ValueError("Window extraction produced empty subarray.")

    # coordinates on the subwindow
    ygrid, xgrid = np.mgrid[0 : sub.shape[0], 0 : sub.shape[1]]

    # estimate background and amplitude from subwindow
    bg = np.median(sub[np.isfinite(sub)])
    amp0 = np.nanmax(sub) - bg
    amp0 = max(amp0, 1.0)

    # initial center guesses in sub-window coords
    if x_guess is None or y_guess is None:
        local_max = np.unravel_index(np.nanargmax(sub), sub.shape)
        x0_local = local_max[1]
        y0_local = local_max[0]
    else:
        x0_local = float(x_guess) - x_min
        y0_local = float(y_guess) - y_min

    g = models.Gaussian2D(
        amplitude=amp0,
        x_mean=x0_local,
        y_mean=y0_local,
        x_stddev=stddev_guess,
        y_stddev=stddev_guess,
    )
    # g = SymmetricGaussian2D(
    #     amplitude=amp0,
    #     x_mean=x0_local,
    #     y_mean=y0_local,
    #     stddev=stddev_guess,
    # )
    c = models.Const2D(amplitude=bg)
    model = g + c  # type: ignore

    # sensible bounds (in local coords)
    g.x_mean.min = 0.0
    g.x_mean.max = sub.shape[1]
    g.y_mean.min = 0.0
    g.y_mean.max = sub.shape[0]
    # g.stddev.min = 0.5
    # g.stddev.max = min(sub.shape) / 2.0
    g.x_stddev.min = 0.5
    g.x_stddev.max = sub.shape[1] / 2.0
    g.y_stddev.min = 0.5
    g.y_stddev.max = sub.shape[0] / 2.0
    g.amplitude.min = 0.0

    mask = np.isfinite(sub)
    fitter = fitting.LevMarLSQFitter()
    fitted = fitter(model, xgrid[mask], ygrid[mask], sub[mask], maxiter=200)
    fit_info = getattr(fitter, "fit_info", None)
    if fit_info is not None:
        fitted.fit_info = fit_info

    fitted.x_mean_0 += x_min  # type: ignore
    fitted.y_mean_0 += y_min  # type: ignore

    return fitted
