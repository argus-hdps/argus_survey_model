"""Spectral sky background model.

Each sky component (airglow, zodiacal, starlight, moonlight, artificial) has
a spectral template, normalized in V, that is integrated through the band's
response to give "e-/s/arcsec^2 per S_10 unit" coefficients. Sky electron
rates are then the dot product of the S_10 values with these coefficients.
"""

from importlib.resources import files
from pathlib import Path
from typing import Iterable, Optional

import astropy.io.ascii as asc
import numpy as np
import scipy.integrate as sing
from .ab_system import ABPhot
from .throughput import SystemThroughput


def _load_template(filepath, wav_nm):
    """Load a spectral template CSV and interpolate onto the throughput grid.

    Template CSVs must have columns ``wav`` (Angstroms) and ``flux``
    (ph/s/um/arcsec^2/m^2 -- photon-counting units). Only the spectral
    *shape* matters here because the template is normalized in V, so units
    cancel. Templates must still be in photon-counting (not energy) units
    because the V-band normalization assumes photon rates; an
    energy-unit template would silently produce wrong color corrections.

    Uses linear interpolation: CubicSpline rings around narrow emission
    lines (e.g. [OI] 557.7nm, OH Meinel bands), producing negative flux
    artifacts and ~5% total flux distortion in the airglow template.

    Raises
    ------
    ValueError
        If the template file does not contain the required ``wav`` and
        ``flux`` columns.

    """
    raw = asc.read(filepath)
    required = {"wav", "flux"}
    actual = set(raw.colnames)
    missing = required - actual
    if missing:
        raise ValueError(
            f"Template {Path(filepath).name} is missing required columns "
            f"{missing}. Expected columns: 'wav' (Angstroms) and 'flux' "
            f"(ph/s/um/arcsec^2/m^2, photon-counting units)."
        )
    raw.sort("wav")
    wav_ang = np.array(raw["wav"])
    flux = np.array(raw["flux"])
    template = np.interp(wav_nm, wav_ang / 10.0, flux, left=0.0, right=0.0)
    return template


class SpectralSky:
    """Precomputed spectral sky model for a single observation band.

    Parameters
    ----------
    throughput : SystemThroughput
        System throughput object (must have V filter loaded).
    ab : ABPhot
        AB photometry object for V-band calibration.
    band : str
        Observation band name (e.g. 'g', 'r', 'i', 'A').
    throughputs_dir : str, optional
        Directory containing spectral template files.

    Attributes
    ----------
    e_per_s10_airglow : float
        Electron rate per S_10 unit for the airglow template [e-/s/arcsec^2].
    e_per_s10_zodiacal : float
        Electron rate per S_10 unit for the zodiacal template [e-/s/arcsec^2].
    e_per_s10_starlight : float
        Electron rate per S_10 unit for the starlight template [e-/s/arcsec^2].
    e_per_s10_moon : float
        Electron rate per S_10 unit for the moonlight template [e-/s/arcsec^2].
    e_per_s10_artif : float
        Electron rate per S_10 unit for the artificial light template [e-/s/arcsec^2].

    """

    def __init__(
        self,
        throughput: SystemThroughput,
        ab: ABPhot,
        band: str,
        throughputs_dir: Optional[str] = None,
    ) -> None:
        """Precompute per-component electron rates from spectral templates."""
        if throughputs_dir is None:
            throughputs_dir = str(files("argus_sim").joinpath("data", "throughputs"))
        tp_dir = Path(throughputs_dir)

        wav = throughput.wav.value  # nm, 300-1000, 10k points

        # --- Per-component dark sky templates ---
        airglow_template = _load_template(str(tp_dir / "airglow_spectrum.csv"), wav)
        zodiacal_template = _load_template(str(tp_dir / "zodiacal_spectrum.csv"), wav)
        starlight_template = _load_template(str(tp_dir / "starlight_spectrum.csv"), wav)

        # Moonlight: scattered solar spectrum from ESO SkyCalc (Noll+ 2012)
        # with Rayleigh + Mie atmospheric scattering included.
        moon_template = _load_template(str(tp_dir / "moonlight_spectrum.csv"), wav)

        # Artificial light: flat spectrum
        artif_template = np.ones_like(wav)

        # system_response leaves out the atmosphere transmission: the SkyCalc
        # templates already include atmospheric absorption at AM=1.2, so
        # applying it again here would double-count extinction.
        system_response = throughput.optics[band] * throughput.filt_wav[band] * throughput.qe_wav

        v_filt = throughput.filt_wav["V"]

        # 1 S_10 = V mag 27.78
        s10_v_photons = ab.photons_from_mag("V", 27.78).value

        templates = {
            "airglow": airglow_template,
            "zodiacal": zodiacal_template,
            "starlight": starlight_template,
            "moon": moon_template,
            "artif": artif_template,
        }

        for name, tmpl in templates.items():
            v_integral = sing.trapezoid(tmpl * v_filt, wav)
            color_correction = sing.trapezoid(tmpl * system_response, wav) / v_integral
            setattr(self, f"e_per_s10_{name}", s10_v_photons * color_correction)

    def sky_electrons(self, airglow_s10, zodiacal_s10, starlight_s10, moon_s10, artif_s10, transparency=1.0):
        """Compute sky electron rate from per-component S_10 values.

        Atmospheric transparency attenuates the three extraterrestrial
        components (airglow, zodiacal, starlight) that must pass through
        the atmosphere.  Moonlight and artificial light are
        scattering-dominated and are not simply attenuated by
        transparency.

        Parameters
        ----------
        airglow_s10 : float or array
            Airglow S_10 value.
        zodiacal_s10 : float or array
            Zodiacal light S_10 value.
        starlight_s10 : float or array
            Starlight S_10 value.
        moon_s10 : float or array
            Moonlight S_10 value.
        artif_s10 : float or array
            Artificial light S_10 value.
        transparency : float
            Atmospheric transparency (0–1).  Applied to airglow,
            zodiacal, and starlight only.  Default 1.0 (no attenuation).

        Returns
        -------
        float or array
            Sky electron rate in e-/s/arcsec^2. Caller multiplies by
            plate_scale**2 for per-pixel rates.

        """
        return (
            self.e_per_s10_airglow * airglow_s10 * transparency
            + self.e_per_s10_zodiacal * zodiacal_s10 * transparency
            + self.e_per_s10_starlight * starlight_s10 * transparency
            + self.e_per_s10_moon * moon_s10
            + self.e_per_s10_artif * artif_s10
        )


def build_band_params(
    bands: Iterable[str],
    throughputs: SystemThroughput,
    ab: ABPhot,
) -> dict[str, dict]:
    """Build per-band throughput and sky electron-rate parameters.

    The signal throughput is tabulated in airmass from the exact photon-rate integral (optics x QE x atmosphere^X
    under the filter); callers evaluate it with :func:`signal_throughput`.  ``signal_tp_no_atm`` (X = 0) and
    ``atm_tp_am1`` (eta(1) / eta(0)) are its summaries, kept for logging and extinction coefficients.

    Parameters
    ----------
    bands : set[str]
        Filter names (e.g. ``{"g", "rho"}``).
    throughputs : SystemThroughput
        System throughput curves.
    ab : ABPhot
        AB photometry system for the throughput.

    Returns
    -------
    dict[str, dict]
        Keyed by band name, each value contains ``signal_tp_table``
        (airmass grid, throughput), ``signal_tp_no_atm``, ``atm_tp_am1``,
        ``band_qe``, and ``sky`` (a `SpectralSky`).

    """
    band_params = {}
    nu = np.asarray(getattr(throughputs.freq, "value", throughputs.freq), dtype=np.float64)
    qe = np.asarray(getattr(throughputs.qe_freq, "value", throughputs.qe_freq), dtype=np.float64)
    atm = np.asarray(throughputs.filt_freq.atmosphere, dtype=np.float64)
    for band in bands:
        resp = np.asarray(throughputs.filt_freq[band], dtype=np.float64)
        optics = float(throughputs.optics[band])
        spectral_sky = SpectralSky(throughputs, ab, band)
        band_qe = np.average(throughputs.qe_freq, weights=throughputs.filt_freq[band])
        qe_b = float(getattr(band_qe, "value", band_qe))
        # Exact signal efficiency of an AB (flat f_nu) source: detected electrons per filter-transmitted photon above
        # the atmosphere, eta(X) = T_opt int R QE T_atm^X dnu/nu / int R dnu/nu (the photon-rate integral with the
        # QE and atmosphere inside it, as SpectralSky integrates the sky; T_opt is the band-averaged optics
        # efficiency).  Factorising it into separate filter-weighted means of QE and atmosphere misses their
        # covariance across the band.  Tabulated in airmass; callers use
        # signal_throughput(), which divides by band_qe because depths are converted with photons = electrons /
        # band_qe against the filter-only AB zero point.
        den = _integrate(resp / nu, nu)
        eta = np.array([_integrate(resp * optics * qe * atm**x / nu, nu) / den for x in _AIRMASS_GRID])
        band_params[band] = {
            "signal_tp_no_atm": float(eta[0] / qe_b),
            "atm_tp_am1": float(np.interp(1.0, _AIRMASS_GRID, eta) / eta[0]),
            "signal_tp_table": (_AIRMASS_GRID, eta / qe_b),
            "band_qe": band_qe,
            "sky": spectral_sky,
        }
    return band_params


_AIRMASS_GRID = np.linspace(0.0, 40.0, 801)


def _integrate(y: np.ndarray, x: np.ndarray) -> float:
    return float(abs(sing.trapezoid(y, x)))


def signal_throughput(bp: dict, airmass) -> np.ndarray | float:
    """Full-system signal throughput at ``airmass`` for one band's parameters from build_band_params.

    This is the exact efficiency eta(X) / band_qe, log-interpolated on the airmass table (Beer-Lambert per
    wavelength, so log eta is near-linear in X).
    """
    x, t = bp["signal_tp_table"]
    return np.exp(np.interp(np.asarray(airmass, dtype=np.float64), x, np.log(t)))
