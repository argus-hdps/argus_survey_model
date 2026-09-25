"""Point spread function models for Argus telescopes."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import astropy.units as u
import numpy as np
from astropy.io import fits
from numba import jit
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

from . import get_logger

# Kolmogorov long-exposure PSF shape parameter
# (Trujillo et al. 2001, MNRAS 328, 977)
MOFFAT_BETA_KOLMOGOROV = 4.765

if TYPE_CHECKING:
    from .config import Config

# Maps simulation bands onto the bands of the delivered PSF file.  The file holds a blue PSF ("g") and a red one
# ("rho"); every red or broad band uses the "rho" PSF.
_BAND_MAP = {"A": "rho", "r": "rho", "i": "rho", "r+i": "rho"}


@jit(nopython=True)
def gaus2d(x: float, y: float, mx: float, my: float, sx: float, sy: float) -> float:
    """Calculate the value of a 2D Gaussian function at given coordinates.

    Parameters
    ----------
    x : float
        The x-coordinate.
    y : float
        The y-coordinate.
    mx : float
        The mean of the Gaussian in the x-direction.
    my : float
        The mean of the Gaussian in the y-direction.
    sx : float
        The standard deviation of the Gaussian in the x-direction.
    sy : float
        The standard deviation of the Gaussian in the y-direction.

    Returns
    -------
    float
        The value of the 2D Gaussian function at the given coordinates.

    """
    return (
        1.0 / (2.0 * np.pi * sx * sy) * np.exp(-((x - mx) ** 2.0 / (2.0 * sx**2.0) + (y - my) ** 2.0 / (2.0 * sy**2.0)))
    )


def _moffat_kernel_2d(fwhm_px: float, beta: float = MOFFAT_BETA_KOLMOGOROV) -> np.ndarray:
    """Return a normalized 2D Moffat kernel for atmospheric seeing."""
    if fwhm_px < 0.01:
        return np.array([[1.0]])
    alpha = (fwhm_px / 2.0) / math.sqrt(2.0 ** (1.0 / beta) - 1.0)
    half = int(math.ceil(6 * alpha)) + 1
    size = 2 * half + 1
    yy, xx = np.mgrid[:size, :size]
    r2 = (xx - half) ** 2 + (yy - half) ** 2
    kernel = (1.0 + r2 / alpha**2) ** (-beta)
    kernel /= kernel.sum()
    return kernel


class GaussianPSF:
    """A class to generate and manipulate Gaussian PSFs.

    Methods
    -------
    with_seeing(seeing, mx=0, my=0, return_fwhm=True)
        Returns downscaled PSF at pixel scale.
    drifted_psf(seeing, rate, angle, exptime, tstep_streak)
        Generate a PSF that accounts for drift during the exposure time.

    """

    def __init__(
        self,
        stamp_size: Optional[int] = 28,  # arcseconds
        stamp_resolution: Optional[float] = 0.1,  # arcseconds
        plate_scale: Optional[float] = 1.007,  # arcseconds/pixel
        pixel_size: Optional[float] = 3.76,  # microns
        rms_spot_size: Optional[float] = 3.2,  # microns
        base_psf: Optional[str] = "gauss",
    ) -> None:
        """Initialize the GaussianPSF object.

        Parameters
        ----------
        stamp_size : Optional[int], optional
            The size of the stamp in arcseconds. Default is 28.
        stamp_resolution : Optional[float], optional
            The resolution of the stamp in arcseconds. Default is 0.1.
        plate_scale : Optional[float], optional
            The plate scale in arcseconds per pixel. Default is 1.007.
        pixel_size : Optional[float], optional
            The size of the pixels in microns. Default is 3.76.
        rms_spot_size : Optional[float], optional
            The RMS spot size in microns. Default is 3.2.
        base_psf : Optional[str], optional
            The base PSF type. Default is "gauss".

        Returns
        -------
        None

        """
        self.log = get_logger(__name__)
        self.stamp_size = stamp_size
        self.stamp_resolution = stamp_resolution
        self.plate_scale = plate_scale
        self.pixel_size = pixel_size
        self.rms_spot_size = rms_spot_size

        x = np.linspace(-stamp_size // 2, stamp_size // 2, np.round(stamp_size / stamp_resolution).astype(int))

        self.x, self.y = np.meshgrid(x, x)

    def with_seeing(
        self, seeing: float, mx: float = 0, my: float = 0, return_fwhm: bool = True, **kwargs
    ) -> np.ndarray:
        """Return downscaled PSF in pixel space.

        Parameters
        ----------
        seeing : float
            The seeing value.
        mx : float, optional
            The mean of the Gaussian in the x-direction. Default is 0.
        my : float, optional
            The mean of the Gaussian in the y-direction. Default is 0.
        return_fwhm : bool, optional
            Whether to return the full width at half maximum (FWHM). Default is True.
        **kwargs
            Ignored. Accepts ``band``, ``rng``, and ``return_optical_hfd``
            for interface compatibility with DeliveredPSF.

        Returns
        -------
        np.ndarray
            The downscaled PSF.
        float, optional
            The FWHM if return_fwhm is True.

        """
        return_optical_hfd = kwargs.get("return_optical_hfd", False)

        optical_stddev = self.plate_scale * (self.rms_spot_size / self.pixel_size)
        psf = gaus2d(self.x, self.y, mx=mx, my=my, sx=optical_stddev, sy=optical_stddev)

        if seeing > 0:
            fwhm_seeing_px = seeing / self.stamp_resolution
            kernel = _moffat_kernel_2d(fwhm_seeing_px)
            psf = fftconvolve(psf, kernel, mode="same")
            psf = np.maximum(psf, 0.0)

        full_res_scale = self.stamp_size / self.x.shape[0]
        real_scale = self.plate_scale
        downscale_factor = int(np.round(real_scale / full_res_scale))

        m_bins = psf.shape[0] // downscale_factor
        n_bins = psf.shape[1] // downscale_factor

        # Crop to exact multiple before reshaping
        psf = psf[: m_bins * downscale_factor, : n_bins * downscale_factor]
        psf = psf.reshape(m_bins, downscale_factor, n_bins, downscale_factor).sum(3).sum(1)
        psf /= psf.sum()
        if return_fwhm:
            delivered_hfd = _half_flux_diameter(psf, real_scale)
            optical_hfd = optical_stddev * 2.355
            if return_optical_hfd:
                return psf, delivered_hfd, optical_hfd
            return psf, delivered_hfd
        return psf

    def drifted_psf(
        self,
        seeing: float,
        rate: u.Quantity[u.arcsec / u.second],
        angle: u.Quantity[u.deg],
        exptime: u.Quantity[u.second],
        tstep_streak: u.Quantity[u.second],
    ) -> np.ndarray:
        """Generate a PSF that accounts for drift during the exposure time.

        Parameters
        ----------
        seeing : float
            The seeing value.
        rate : u.Quantity[u.arcsec / u.second]
            The drift rate in arcseconds per second.
        angle : u.Quantity[u.deg]
            The drift angle in degrees.
        exptime : u.Quantity[u.second]
            The exposure time in seconds.
        tstep_streak : u.Quantity[u.second]
            The time step for the streak in seconds.

        Returns
        -------
        np.ndarray
            The drifted PSF.

        """
        timesteps = np.arange(0, exptime.to(u.second).value, tstep_streak.to(u.second).value) * u.second
        distance = rate * timesteps
        y = np.sin(angle.to("radian")) * distance
        x = np.cos(angle.to("radian")) * distance
        if (np.abs(x).max() > self.stamp_size * u.arcsecond / 2) or (
            np.abs(y).max() > self.stamp_size * u.arcsecond / 2
        ):
            self.log.warning("Streak too long for stamp size.")

        self.log.debug(f"Streaked length: {np.sqrt((y[-1] - y[0]) ** 2 + (x[-1] - x[0]) ** 2)}")
        self.log.debug(f"Integrating streak over {len(timesteps)} timesteps")
        base_psf = self.with_seeing(seeing, return_fwhm=False)
        streaked_psf = np.zeros((base_psf.shape[0], base_psf.shape[1], len(timesteps)))
        streaked_psf[:, :, 0] = base_psf
        for i in range(1, len(timesteps)):
            streaked_psf[:, :, i] = self.with_seeing(seeing, mx=x[i], my=y[i], return_fwhm=False)
        psf = streaked_psf.sum(axis=-1)
        psf /= psf.sum()
        return psf


def _line_kernel(length_px: float, angle_deg: float) -> np.ndarray:
    """Create a normalized uniform line kernel.

    Parameters
    ----------
    length_px : float
        Trail length in pixels.
    angle_deg : float
        Position angle in degrees, measured counterclockwise from the
        x-axis (column direction).

    Returns
    -------
    np.ndarray
        Normalized 2D kernel with flux distributed uniformly along the line.

    """
    if length_px < 0.1:
        return np.array([[1.0]])

    half = int(math.ceil(length_px / 2)) + 1
    size = 2 * half + 1
    kernel = np.zeros((size, size))

    n_samples = max(int(math.ceil(length_px * 10)), 50)
    t = np.linspace(-length_px / 2, length_px / 2, n_samples)
    angle_rad = math.radians(angle_deg)
    dx = t * math.cos(angle_rad)
    dy = t * math.sin(angle_rad)

    for x, y in zip(dx, dy):
        ix = x + half
        iy = y + half
        x0 = int(math.floor(ix))
        y0 = int(math.floor(iy))
        fx = ix - x0
        fy = iy - y0
        if 0 <= x0 < size - 1 and 0 <= y0 < size - 1:
            kernel[y0, x0] += (1 - fx) * (1 - fy)
            kernel[y0, x0 + 1] += fx * (1 - fy)
            kernel[y0 + 1, x0] += (1 - fx) * fy
            kernel[y0 + 1, x0 + 1] += fx * fy

    kernel /= kernel.sum()
    return kernel


def elongate_psf(
    psf: np.ndarray,
    elongation_arcsec: float,
    angle_deg: float,
    plate_scale: float,
) -> np.ndarray:
    """Convolve a PSF stamp with a uniform line kernel.

    Models the PSF elongation from tracking drift (e.g. differential
    atmospheric refraction across a monolithic tracking mount).  The
    trail is a uniform line of the specified length at the given
    position angle, convolved into the existing PSF stamp.

    Parameters
    ----------
    psf : np.ndarray
        Normalized 2D PSF stamp (pixel scale).
    elongation_arcsec : float
        Total trail length in arcseconds.
    angle_deg : float
        Position angle of the trail in degrees, measured
        counterclockwise from the x-axis (column direction).
    plate_scale : float
        Arcseconds per pixel.

    Returns
    -------
    np.ndarray
        Elongated PSF stamp, same shape as input, normalized to sum=1.

    """
    length_px = elongation_arcsec / plate_scale
    if length_px < 0.1:
        return psf

    kernel = _line_kernel(length_px, angle_deg)
    from scipy.signal import fftconvolve

    convolved = fftconvolve(psf, kernel, mode="same")
    convolved /= convolved.sum()
    return convolved


def _half_flux_diameter(psf: np.ndarray, spacing: float) -> float:
    """Compute the half-flux diameter of a 2D PSF array.

    Parameters
    ----------
    psf : np.ndarray
        Normalized 2D PSF.
    spacing : float
        Pixel spacing in physical units (e.g. microns). The returned HFD
        is in the same units.

    Returns
    -------
    float
        Diameter encircling half the total flux.

    """
    cy, cx = np.unravel_index(np.argmax(psf), psf.shape)
    yy, xx = np.mgrid[: psf.shape[0], : psf.shape[1]]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) * spacing
    total = psf.sum()

    r_flat = r.ravel()
    flux_flat = psf.ravel()
    half_px = 0.5 * spacing
    r_eval = np.linspace(0, r_flat.max(), 200)
    ee = np.array([np.sum(flux_flat * np.clip((rt - r_flat + half_px) / spacing, 0, 1)) for rt in r_eval])
    r_half = np.interp(0.5 * total, ee, r_eval)
    return 2.0 * float(r_half)


def _match_ee50(
    psf: np.ndarray,
    spacing: float,
    target_hfd: float,
    tol: float = 0.01,
    max_iter: int = 20,
) -> np.ndarray:
    """Convolve *psf* with a Gaussian kernel sized to hit a target HFD.

    The naive closed-form kernel (assuming Gaussian HFD scaling) undersizes
    the kernel for non-Gaussian PSFs such as a diffraction pattern
    with side lobes.  This function uses bisection on the kernel sigma to match the
    measured half-flux diameter to ``target_hfd`` within fractional
    tolerance ``tol``.

    Parameters
    ----------
    psf : np.ndarray
        Input PSF (normalized, fine-sampled).
    spacing : float
        Pixel spacing in physical units (microns).
    target_hfd : float
        Desired half-flux diameter in the same units as *spacing*.
    tol : float
        Fractional convergence tolerance on HFD.
    max_iter : int
        Maximum bisection iterations.

    Returns
    -------
    np.ndarray
        Convolved PSF with HFD matching ``target_hfd``.

    """
    base_hfd = _half_flux_diameter(psf, spacing)
    if target_hfd <= base_hfd:
        return psf

    # Gaussian initial guess: sigma = D * sqrt(mult² - 1) / 2.355
    mult = target_hfd / base_hfd
    sigma_guess = base_hfd * math.sqrt(mult**2 - 1) / 2.355 / spacing

    # Bracket: start from a small positive value to avoid zero,
    # and use 4× the guess as upper bound for non-Gaussian PSFs.
    lo, hi = sigma_guess * 0.1, sigma_guess * 4.0

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        convolved = gaussian_filter(psf, sigma=mid)
        measured = _half_flux_diameter(convolved, spacing)
        if abs(measured - target_hfd) / target_hfd < tol:
            return convolved
        if measured < target_hfd:
            lo = mid
        else:
            hi = mid

    return gaussian_filter(psf, sigma=(lo + hi) / 2.0)


def load_ee50_profile(path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read an EE50 profile CSV with columns ``band, theta_deg, target_um``.

    ``band`` is a PSF-file band name (``g``, ``rho``).  ``target_um`` is the optics-only EE50 diameter in microns
    wanted at field angle ``theta_deg``.  Rows are sorted by field angle per band; ``DeliveredPSF.with_seeing``
    interpolates linearly in angle.
    """
    import csv

    rows: dict[str, list[tuple[float, float]]] = {}
    with open(path) as f:
        for rec in csv.DictReader(f):
            rows.setdefault(rec["band"], []).append((float(rec["theta_deg"]), float(rec["target_um"])))
    out = {}
    for band, pts in rows.items():
        pts.sort()
        out[band] = (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    return out


class DeliveredPSF:
    """Field-dependent PSF of the delivered telescope optics, tabulated at a set of field angles.

    Loads a FITS binary table of optics-only PSF images (no atmosphere), one per band and field angle, sampled
    ``OVERSAMP`` times finer than the detector pixel.  Each call convolves atmospheric seeing, bins to detector
    pixels at a chosen sub-pixel phase and cuts a stamp.

    The file has one binary-table extension with columns ``BAND`` (the PSF band, ``g`` or ``rho``), ``THETA``
    (field angle from the optical axis, degrees) and ``PSF`` (the ``NPIX x NPIX`` image, flattened), and header
    keywords ``OVERSAMP`` and ``NPIX``.

    Parameters
    ----------
    fits_path : str or Path
        Path to the PSF FITS file.
    plate_scale : float
        Arcseconds per pixel.
    pixel_size : float
        Detector pixel size in microns.
    n_pixels_x : int
        Detector long-axis pixel count.
    n_pixels_y : int
        Detector short-axis pixel count.
    stamp_size : int
        Output stamp size in pixels.
    ee50_profile : dict, optional
        Optics-only EE50 target per PSF band: ``{band: (theta_deg, target_um)}``, see :func:`load_ee50_profile`.
        Where the target exceeds the tabulated PSF's EE50 diameter the PSF is broadened with a Gaussian to meet it;
        a smaller target leaves the PSF unchanged.
    randomize_pixel_phase : bool
        Whether to apply random subpixel phase shift during downsampling.

    """

    def __init__(
        self,
        fits_path: str | Path,
        plate_scale: float,
        pixel_size: float,
        n_pixels_x: int = 11648,
        n_pixels_y: int = 8742,
        stamp_size: int = 28,
        ee50_profile: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
        randomize_pixel_phase: bool = False,
    ) -> None:
        """Load the PSF table and build a per-band field angle index."""
        self.log = get_logger(__name__)
        self.plate_scale = plate_scale
        self.pixel_size = pixel_size
        self.n_pixels_x = n_pixels_x
        self.n_pixels_y = n_pixels_y
        self.stamp_size = stamp_size
        self.ee50_profile = ee50_profile
        self.randomize_pixel_phase = randomize_pixel_phase

        with fits.open(fits_path) as hdul:
            header = hdul[1].header
            table = hdul[1].data
            self._oversample = int(header["OVERSAMP"])
            npix = int(header["NPIX"])
            bands = np.array([b.strip() for b in table["BAND"]])
            field_angles = np.array(table["THETA"], dtype=np.float64)
            self._images = np.array(table["PSF"], dtype=np.float64).reshape(len(bands), npix, npix)
        self._images /= self._images.sum(axis=(1, 2), keepdims=True)
        self._spacing = pixel_size / self._oversample

        self._band_index: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for band_str in np.unique(bands):
            rows = np.where(bands == band_str)[0]
            order = np.argsort(field_angles[rows])
            self._band_index[band_str] = (field_angles[rows][order], rows[order])

    def field_angles(self, band: str) -> np.ndarray:
        """Tabulated field angles (degrees, increasing) for ``band``."""
        return np.asarray(self._band_index[_BAND_MAP.get(band, band)][0], dtype=np.float64)

    @classmethod
    def from_config(cls, cfg: Config, **overrides) -> DeliveredPSF:
        """Build a DeliveredPSF from a Config object.

        Parameters
        ----------
        cfg : Config
            Simulation configuration.
        **overrides
            Keyword arguments forwarded to the constructor, overriding
            the config-derived values.

        """
        if cfg.telescope.psf_fits_path is None:
            raise ValueError("psf_model='delivered' requires psf_fits_path")
        kwargs = dict(
            fits_path=cfg.telescope.psf_fits_path,
            plate_scale=cfg.telescope.plate_scale,
            pixel_size=cfg.telescope.pixel_size,
            n_pixels_x=cfg.telescope.n_pixels_x,
            n_pixels_y=cfg.telescope.n_pixels_y,
            stamp_size=cfg.survey.stamp_size,
            ee50_profile=load_ee50_profile(cfg.telescope.ee50_profile_path)
            if cfg.telescope.ee50_profile_path
            else None,
            randomize_pixel_phase=cfg.telescope.randomize_pixel_phase,
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    def with_seeing(
        self,
        seeing: float,
        band: str = "g",
        rng: np.random.Generator | None = None,
        pixel_phase: Optional[tuple[float, float]] = None,
        return_optical_hfd: bool = False,
        field_angle_deg: float | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, float]:
        """Generate a PSF stamp with atmospheric seeing.

        Parameters
        ----------
        seeing : float
            Atmospheric seeing FWHM in arcseconds.
        band : str
            Simulation band name (mapped to a PSF band internally).
        rng : np.random.Generator, optional
            Random generator for the field position and pixel phase draws.
        pixel_phase : tuple[float, float], optional
            Subpixel phase offset ``(phase_y, phase_x)`` each in ``[0, 1)``,
            controlling where the detector pixel grid falls relative to the
            finely sampled optical PSF.  ``(0, 0)`` means the binning grid is
            aligned with the array origin; ``(0.5, 0.5)`` shifts by half a
            detector pixel in both axes.  When ``None`` (default) and
            ``randomize_pixel_phase`` is enabled, a uniform random phase is
            drawn; when ``None`` and randomization is disabled, ``(0, 0)`` is
            used.
        return_optical_hfd : bool
            If True, also return the optics-only HFD (before seeing
            convolution, in microns) as a third element.
        field_angle_deg : float, optional
            Field angle (degrees from the OTA's optical axis) to evaluate at: the
            nearest tabulated angle is used.  When None (default) a random
            detector position is drawn from ``rng``.
        **kwargs
            Ignored; accepted for interface compatibility.

        Returns
        -------
        tuple[np.ndarray, float] or tuple[np.ndarray, float, float]
            ``(stamp, hfd)``: normalized PSF stamp and half-flux diameter
            in arcseconds.  When ``return_optical_hfd`` is True, returns
            ``(stamp, hfd, optical_hfd)`` where ``optical_hfd`` is the
            HFD before atmospheric seeing convolution.

        """
        _rng = rng if rng is not None else np.random.default_rng()

        psf_band = _BAND_MAP.get(band, band)
        if psf_band not in self._band_index:
            raise ValueError(f"Band '{band}' (mapped to '{psf_band}') not in the PSF file")
        angles, rows = self._band_index[psf_band]

        if field_angle_deg is None:
            # Draw a random field position and find the nearest tabulated angle
            x_pix = _rng.uniform(-self.n_pixels_x / 2, self.n_pixels_x / 2)
            y_pix = _rng.uniform(-self.n_pixels_y / 2, self.n_pixels_y / 2)
            theta_deg = math.sqrt(x_pix**2 + y_pix**2) * self.plate_scale / 3600.0
        else:
            theta_deg = float(field_angle_deg)
        nearest_idx = int(np.argmin(np.abs(angles - theta_deg)))
        psf = self._images[rows[nearest_idx]].copy()
        spacing = self._spacing
        ds = self._oversample

        profile = self.ee50_profile.get(psf_band) if self.ee50_profile else None
        if profile is not None:
            target_hfd = float(np.interp(float(angles[nearest_idx]), profile[0], profile[1]))
            psf = _match_ee50(psf, spacing, target_hfd=target_hfd)

        optical_hfd = _half_flux_diameter(psf, spacing) if return_optical_hfd else None

        if seeing > 0:
            fwhm_seeing_um = seeing * (self.pixel_size / self.plate_scale)
            fwhm_seeing_px = fwhm_seeing_um / spacing
            kernel = _moffat_kernel_2d(fwhm_seeing_px)
            psf = fftconvolve(psf, kernel, mode="same")
            psf = np.maximum(psf, 0.0)
            psf /= psf.sum()

        if pixel_phase is not None:
            offset_y = int(pixel_phase[0] * ds)
            offset_x = int(pixel_phase[1] * ds)
        elif self.randomize_pixel_phase:
            offset_y = int(_rng.uniform(0, ds))
            offset_x = int(_rng.uniform(0, ds))
        else:
            offset_y = 0
            offset_x = 0

        psf = psf[offset_y:, offset_x:]
        ny = psf.shape[0] // ds
        nx = psf.shape[1] // ds
        psf = psf[: ny * ds, : nx * ds]
        psf = psf.reshape(ny, ds, nx, ds).sum(axis=(1, 3))

        cy, cx = np.unravel_index(np.argmax(psf), psf.shape)
        half = self.stamp_size // 2
        y0 = max(0, cy - half)
        x0 = max(0, cx - half)
        y1 = min(psf.shape[0], y0 + self.stamp_size)
        x1 = min(psf.shape[1], x0 + self.stamp_size)
        y0 = y1 - self.stamp_size
        x0 = x1 - self.stamp_size
        stamp = psf[y0:y1, x0:x1]

        stamp /= stamp.sum()

        hfd = _half_flux_diameter(stamp, self.plate_scale)

        if return_optical_hfd:
            return stamp, hfd, optical_hfd
        return stamp, hfd
