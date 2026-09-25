"""Forward Monte Carlo uncertainty propagation for ArgusSim.

Draws samples from prior distributions on uncertain input parameters
(read noise, dark current, throughput, solar activity, seeing, transparency), runs a 36-ratchet
representative grid for each draw, and analytically scales to survey
totals. Produces depth distributions for sensitivity analysis.

Example usage::

    from argus_sim.config import Config
    from argus_sim.forward_mc import ForwardMC

    cfg = Config(sim_version="1.0")
    mc = ForwardMC(cfg, output_dir="output_mc")
    results = mc.run(n_samples=100, seed=42)

    df = mc.to_dataframe()

"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import datetime
from pathlib import Path

import astropy.time as atime
import astropy.units as u
import numpy as np

from .ab_system import ABPhot
from .config import Config
from .observatory import Observatory, seasonal_seeing_params, seeing_at_airmass, site_seeing, site_weather
from .provenance import write_provenance
from .photon_budget import NoiseBudget
from .psf import GaussianPSF, DeliveredPSF, elongate_psf
from .spectral_sky import build_band_params, signal_throughput
from .throughput import SystemThroughput

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Monthly weather statistics derived from site_weather lookup table
# ---------------------------------------------------------------------------

# Each month has 3 weekly clear-fraction measurements (days 1, 11, 21).
# μ_month = mean of the 3 weeks; σ_month = std of the 3 weeks.
MONTHLY_WEATHER_STATS: dict[int, tuple[float, float]] = {}
for _mo in range(1, 13):
    _vals = np.array(list(site_weather[_mo].values()))
    MONTHLY_WEATHER_STATS[_mo] = (float(_vals.mean()), float(_vals.std()))
del _mo, _vals

LUNAR_MODES = ("dark", "base", "bright")


# ---------------------------------------------------------------------------
# Prior distribution
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Prior:
    """A single parameter prior distribution.

    Parameters
    ----------
    field : str
        Dot-separated config path, e.g. ``"telescope.readnoise"``.
    dist : str
        Distribution type: ``"normal"``, ``"lognormal"``, ``"uniform"``,
        ``"invgamma"``, ``"burr12"``, ``"empirical"`` (uniform draw
        from ``{"values": [...]}``), ``"cloud_lognormal"`` (a
        transparency: see :func:`draw_cloud_transparency`), or
        ``"seasonal_lognormal"`` (the annual marginal of the site's
        seasonal seeing).
    params : dict
        Distribution parameters. For ``"normal"``: ``{"mean", "std"}``.
        For ``"lognormal"``: ``{"mean", "std"}`` (of the underlying
        normal). For ``"uniform"``: ``{"low", "high"}``.
        For ``"invgamma"``: ``{"a", "loc", "scale"}`` (scipy parameterization).
        For ``"burr12"``: ``{"c", "d", "loc", "scale"}`` (scipy parameterization).
    clip : tuple, optional
        ``(low, high)`` bounds to clip the drawn value. Either bound
        may be ``None`` for one-sided clipping.

    """

    field: str
    dist: str
    params: dict
    clip: tuple[float | None, float | None] | None = None

    def draw(self, rng: np.random.Generator) -> float:
        """Draw a single sample from this prior distribution."""
        if self.dist == "normal":
            value = float(rng.normal(self.params["mean"], self.params["std"]))
        elif self.dist == "lognormal":
            value = float(rng.lognormal(self.params["mean"], self.params["std"]))
        elif self.dist == "empirical":
            return float(rng.choice(np.asarray(self.params["values"], dtype=np.float64)))
        elif self.dist == "cloud_lognormal":
            return draw_cloud_transparency(rng, **self.params)
        elif self.dist == "seasonal_lognormal":
            # the annual marginal of the per-epoch seasonal draw (draw_seasonal_seeing): a winter month with
            # probability winter_fraction, else the rest-of-year season
            winter = rng.random() < self.params["winter_fraction"]
            med, sig = (
                (self.params["winter"], self.params["winter_sigma"])
                if winter
                else (self.params["rest"], self.params["rest_sigma"])
            )
            return float(rng.lognormal(np.log(med), sig))
        elif self.dist == "uniform":
            value = float(rng.uniform(self.params["low"], self.params["high"]))
        elif self.dist == "invgamma":
            from scipy import stats

            value = float(
                stats.invgamma.rvs(
                    self.params["a"],
                    loc=self.params["loc"],
                    scale=self.params["scale"],
                    random_state=rng,
                )
            )
        elif self.dist == "burr12":
            from scipy import stats

            value = float(
                stats.burr12.rvs(
                    self.params["c"],
                    self.params["d"],
                    loc=self.params["loc"],
                    scale=self.params["scale"],
                    random_state=rng,
                )
            )
        else:
            raise ValueError(f"Unknown distribution: {self.dist}")
        if self.clip is not None:
            lo = self.clip[0] if self.clip[0] is not None else -np.inf
            hi = self.clip[1] if self.clip[1] is not None else np.inf
            value = float(np.clip(value, lo, hi))
        return value


# ---------------------------------------------------------------------------
# Default priors
# ---------------------------------------------------------------------------

# Solar-activity index over the planned operating epoch, one value per month
# 2028-01 .. 2033-12 (72 values), used as an empirical prior.  Walker (1987) /
# ING S_sun runs 0.8 at solar minimum to 2.0 at maximum; mapped linearly onto
# the 13-month smoothed sunspot number, s_sun = 0.8 + 1.2 * SSN / 160.9, with
# 160.9 the cycle-25 smoothed maximum (NOAA SWPC observed indices, 2024-10).
# SSN: NOAA SWPC predicted-solar-cycle.json (fetched 2026-09-06) through
# 2030-12; an assumed minimum of 8 at 2031-03; then a Hathaway (2015) shape
# rising to a cycle-26 maximum equal to cycle 25's, 4 yr after minimum.
# Mean 1.10 (airglow 178 S10).
SOLAR_ACTIVITY_2028_2033: list[float] = [
    1.2393,
    1.2206,
    1.2042,
    1.1863,
    1.1699,
    1.1535,
    1.1371,
    1.1214,
    1.1065,
    1.0916,
    1.0774,
    1.064,
    1.0506,
    1.0372,
    1.026,
    1.014,
    1.0029,
    0.9917,
    0.9812,
    0.9708,
    0.9611,
    0.9521,
    0.9424,
    0.9342,
    0.926,
    0.9186,
    0.9119,
    0.9044,
    0.8984,
    0.8917,
    0.8858,
    0.8798,
    0.8746,
    0.8701,
    0.8649,
    0.8604,
    0.8601,
    0.8598,
    0.8597,
    0.86,
    0.861,
    0.8634,
    0.8675,
    0.8738,
    0.8825,
    0.8939,
    0.9082,
    0.9256,
    0.9459,
    0.9692,
    0.9954,
    1.0243,
    1.0557,
    1.0894,
    1.1251,
    1.1625,
    1.2014,
    1.2414,
    1.2823,
    1.3238,
    1.3655,
    1.4073,
    1.4489,
    1.49,
    1.5305,
    1.5701,
    1.6086,
    1.6459,
    1.6818,
    1.7162,
    1.7489,
    1.7799,
]

# Per-sample priors: fixed for a given MC draw (array/environment properties).
# Read noise and dark current use camera-level priors: the central value is the
# per-pixel median from the IMX455 array (2025 lab campaign), and the spread
# represents camera-to-camera variability (±10% default, configurable via
# telescope.readnoise_variability and telescope.dark_current_variability).
# The per-pixel distributions are wider (Burr XII for readnoise, inverse-gamma
# for dark current).
# Optics throughput is fit to flat field percentiles (captures vignetting,
# filter, and QE variation).
DEFAULT_SAMPLE_PRIORS = [
    Prior("telescope.readnoise", "normal", {"mean": 1.08, "std": 0.108}, clip=(0.0, None)),
    Prior("telescope.dark_current", "normal", {"mean": 0.00063, "std": 0.000063}, clip=(0.0, None)),
    Prior("telescope.throughput_loss", "normal", {"mean": 0.874, "std": 0.051}, clip=(0.5, 1.0)),
    Prior("observatory.solar_activity", "empirical", {"values": SOLAR_ACTIVITY_2028_2033}),
]

# Grey cloud/cirrus loss over clear nights (clear nights themselves come from the weather model,
# observatory.site_weather).  A fraction of exposures is photometric (no extra loss): 0.78, DES Y3 FGCM at CTIO
# (Burke et al. 2018, 2018AJ....155...41B: 77.7% of griz exposures are photometric calibration exposures).  The rest
# lose extra ~ lognormal(median 0.378 mag, sigma_ln 1.143), clipped at 2 mag, fitted through the
# throughput tail of Gebhardt et al. (2021, 2021ApJ...923..217G, Fig. tp4940: ~6.5% of observations > 0.7 mag and
# ~2.5% > 1.5 mag below the best-conditions throughput; an upper bound, as it includes fibre/seeing losses).
CLOUD_TRANSPARENCY = {"photometric_fraction": 0.78, "median_mag": 0.378, "sigma_ln": 1.143, "max_mag": 2.0}

# Per-ratchet priors: drawn fresh for each representative ratchet, as draw_atmosphere draws them.  The seeing follows the DIMM seasons of observatory.site_seeing (Barker et al. 2003), drawn per
# epoch; this entry is its annual marginal (3 winter months in 12).
SEASONAL_SEEING_PRIOR = {
    "winter": site_seeing[1]["median"],
    "winter_sigma": site_seeing[1]["std"],
    "rest": site_seeing[6]["median"],
    "rest_sigma": site_seeing[6]["std"],
    "winter_fraction": 3 / 12,
}

DEFAULT_RATCHET_PRIORS = [
    Prior("observatory.seeing_mean", "seasonal_lognormal", dict(SEASONAL_SEEING_PRIOR)),
    Prior("observatory.transparency", "cloud_lognormal", dict(CLOUD_TRANSPARENCY)),
]

# Sample and ratchet priors in one list
DEFAULT_PRIORS = DEFAULT_SAMPLE_PRIORS + DEFAULT_RATCHET_PRIORS


# ---------------------------------------------------------------------------
# Config field helpers
# ---------------------------------------------------------------------------


def _set_config_field(config: Config, dotpath: str, value: float) -> None:
    """Set a nested config field via dot-separated path."""
    parts = dotpath.split(".")
    obj = config
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _get_config_field(config: Config, dotpath: str) -> float:
    """Get a nested config field via dot-separated path."""
    obj = config
    for part in dotpath.split("."):
        obj = getattr(obj, part)
    return obj


def perturb_config(
    base_config: Config,
    priors: list[Prior],
    rng: np.random.Generator,
) -> tuple[Config, dict[str, float]]:
    """Create a perturbed copy of the config by drawing from priors.

    Returns the new config and a dict of the drawn parameter values.
    """
    cfg = dataclasses.replace(
        base_config,
        observatory=dataclasses.replace(base_config.observatory),
        telescope=dataclasses.replace(base_config.telescope),
        survey=dataclasses.replace(base_config.survey),
        uptime=dataclasses.replace(base_config.uptime),
        output=dataclasses.replace(base_config.output),
    )

    theta = {}
    for prior in priors:
        value = prior.draw(rng)
        _set_config_field(cfg, prior.field, value)
        theta[prior.field] = value

    return cfg, theta


# ---------------------------------------------------------------------------
# Representative ratchet grid
# ---------------------------------------------------------------------------

# Representative dates anchored to actual new moons in 2026 so that
# offsets [0, 7, 14, 21] days sample the lunar cycle correctly:
#   +0d ≈ new moon (illum ≈ 0), +7d ≈ first quarter (≈ 0.5),
#   +14d ≈ full moon (≈ 1.0), +21d ≈ third quarter (≈ 0.5–0.6).
_SEASON_DATES = {
    "winter": datetime(2026, 1, 19),
    "equinox": datetime(2026, 3, 19),
    "summer": datetime(2026, 6, 15),
}

# Lunar phase offsets from start of each season date (days).
# The actual illumination fraction is computed from ephemeris.
_LUNAR_OFFSETS_DAYS = [0, 7, 14, 21]

# Representative altitudes → airmasses
_REPR_ALTITUDES_DEG = [90.0, 56.4, 38.7]  # → airmass ~1.0, ~1.2, ~1.6
_REPR_AIRMASSES = [1.0, 1.2, 1.6]

# Night lengths at the representative site latitude for
# astronomical twilight (sun < -18°), in hours.
_SEASON_NIGHT_HOURS = {
    "winter": 11.0,
    "equinox": 8.5,
    "summer": 6.0,
}


@dataclasses.dataclass
class RatchetGridPoint:
    """One point in the representative ratchet grid."""

    season: str
    lunar_offset_days: int
    alt_deg: float
    airmass: float
    time: atime.Time
    lunar_illumination: float = 0.0


def _compute_lunar_illumination(t: atime.Time) -> float:
    """Compute lunar illumination fraction at a given time."""
    import astroplan

    return float(astroplan.moon_illumination(t))


def build_ratchet_grid(
    timezone_offset_hours: float = -6.0,
    lunar_mode: str = "base",
    bright_threshold: float = 0.93,
) -> list[RatchetGridPoint]:
    """Build the representative ratchet grid for a given lunar mode.

    Parameters
    ----------
    timezone_offset_hours : float
        UTC offset for local midnight (default -6 for CST).
    lunar_mode : str
        One of ``"dark"``, ``"base"``, ``"bright"``.

        - **dark**: Only new-moon grid points (offset=0, illumination ≈ 0).
          Yields 9 points (1 × 3 seasons × 3 airmasses).
        - **base**: Illumination below ``bright_threshold``.
        - **bright**: Illumination at or above ``bright_threshold``.
    bright_threshold : float
        Illumination fraction that separates base cadence from bright
        time.  Default 0.93.

    Returns
    -------
    list[RatchetGridPoint]
        Grid points matching the requested lunar mode.

    """
    if lunar_mode not in LUNAR_MODES:
        raise ValueError(f"Unknown lunar_mode={lunar_mode!r}, expected one of {LUNAR_MODES}")

    grid = []
    for season, base_date in _SEASON_DATES.items():
        for offset_days in _LUNAR_OFFSETS_DAYS:
            dt = base_date.replace(hour=0, minute=0, second=0)
            utc_hour = int(-timezone_offset_hours)
            t = atime.Time(datetime(dt.year, dt.month, dt.day, utc_hour, 0, 0)) + offset_days * u.day
            illum = _compute_lunar_illumination(t)

            if lunar_mode == "dark" and offset_days != 0:
                continue
            elif lunar_mode == "base" and illum >= bright_threshold:
                continue
            elif lunar_mode == "bright" and illum < bright_threshold:
                continue

            for alt_deg, airmass in zip(_REPR_ALTITUDES_DEG, _REPR_AIRMASSES):
                grid.append(
                    RatchetGridPoint(
                        season=season,
                        lunar_offset_days=offset_days,
                        alt_deg=alt_deg,
                        airmass=airmass,
                        time=t,
                        lunar_illumination=illum,
                    )
                )
    return grid


def compute_grid_weights(
    grid: list[RatchetGridPoint],
    airmass_fractions: dict[float, float] | None = None,
) -> np.ndarray:
    """Compute the fractional weight of each grid point.

    The weight is the product of:
    - Season weight: proportional to night length
    - Lunar phase weight: uniform (each quarter ~25%)
    - Airmass weight: fraction of footprint at each airmass

    Parameters
    ----------
    grid : list[RatchetGridPoint]
        The 36-point grid.
    airmass_fractions : dict[float, float], optional
        Fraction of footprint at each representative airmass.  Default
        {1.0: 0.11, 1.2: 0.57, 1.6: 0.32}, from the ring layout.

    Returns
    -------
    np.ndarray
        Normalized weight array (sums to 1.0).

    """
    if airmass_fractions is None:
        # HEALPix solid-angle-weighted fractions of the ring-layout footprint
        # at each representative airmass, allowing for FoV overlap in the
        # dense zenith cap.
        airmass_fractions = {1.0: 0.11, 1.2: 0.57, 1.6: 0.32}

    total_night_hours = sum(_SEASON_NIGHT_HOURS.values())
    n_lunar_phases = len({pt.lunar_offset_days for pt in grid})

    weights = np.zeros(len(grid))
    for i, pt in enumerate(grid):
        w_season = _SEASON_NIGHT_HOURS[pt.season] / total_night_hours
        w_lunar = 1.0 / n_lunar_phases
        w_airmass = airmass_fractions.get(pt.airmass, 1.0 / len(_REPR_AIRMASSES))
        weights[i] = w_season * w_lunar * w_airmass

    weights /= weights.sum()
    return weights


# ---------------------------------------------------------------------------
# Per-month weather draws
# ---------------------------------------------------------------------------


def draw_monthly_weather(
    rng: np.random.Generator,
    stats: dict[int, tuple[float, float]] | None = None,
) -> dict[int, float]:
    """Draw a 12-element per-month clear-fraction vector.

    Each month is drawn from N(μ_month, σ_month) where μ and σ come
    from the weekly site weather data. Values are clipped to [0, 1].

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    stats : dict, optional
        Mapping of month (1–12) to (mean, std). Defaults to
        :data:`MONTHLY_WEATHER_STATS`.

    Returns
    -------
    dict[int, float]
        Month (1–12) → clear fraction for this MC sample.

    """
    if stats is None:
        stats = MONTHLY_WEATHER_STATS
    weather = {}
    for month in range(1, 13):
        mu, sigma = stats[month]
        val = float(rng.normal(mu, sigma))
        weather[month] = float(np.clip(val, 0.0, 1.0))
    return weather


def _annual_mean_clear_fraction(monthly_weather: dict[int, float]) -> float:
    """Compute the exposure-weighted annual mean clear fraction.

    Weights months by their night length (winter months contribute
    more observing time).
    """
    month_to_season = {
        12: "winter",
        1: "winter",
        2: "winter",
        3: "equinox",
        4: "equinox",
        5: "equinox",
        6: "summer",
        7: "summer",
        8: "summer",
        9: "equinox",
        10: "equinox",
        11: "equinox",
    }
    total_weight = 0.0
    weighted_sum = 0.0
    for month, f_clear in monthly_weather.items():
        season = month_to_season[month]
        night_h = _SEASON_NIGHT_HOURS[season]
        total_weight += night_h
        weighted_sum += night_h * f_clear
    return weighted_sum / total_weight


# ---------------------------------------------------------------------------
# Cadence windows
# ---------------------------------------------------------------------------

# "1sec" is one exposure whatever its length; with 60 s exposures (dark and
# base lunar modes) it therefore equals "1min", and the two columns of the
# summary CSV are identical by construction.  They differ only in bright
# mode, where exposures are 1 s.
DEFAULT_CADENCE_WINDOWS = ("1sec", "1min", "15min", "30min", "1hour", "1night", "1week", "6month", "5year")

_NIGHTS_PER_YEAR = 365.0


#: How a cadence window's exposures are split between the bands of the filter cycle.
#: ``"block"`` (the default) treats the cycle as strictly exchanged
#: 15-exposure blocks (the exposures of one 16-minute pointing) with no
#: camera overlap.
DUTY_MODELS = ("block",)
BLOCK_MINUTES = 15.0


def block_duty(
    options: list[str], n_all: float, exptime_s: float, block_minutes: float = BLOCK_MINUTES
) -> dict[str, dict[str, float]]:
    """Per-band on-duty exposure counts for a window under strict block alternation.

    The filter cycle is taken as exchanged pointings:
    every sky position sees ``block_minutes`` in each entry of ``options`` in
    turn, with no camera overlap: ``["g", "rho", "g", "rho"]`` is one
    15-exposure block of beta, one of rho, repeating; ``["g", "g", "rho"]``
    is two blocks of beta then one of rho.  A block is the 15 exposures of
    one 16-minute pointing.
    A cadence window starts on a block boundary, and its starting block is
    uniform over the cycle.  For a window of ``n_all`` exposures (all bands)
    the on-duty count in a band is the number of exposures in that band's
    blocks inside the window; ``f_on`` is the fraction of starting phases in
    which the band appears at all.

    So a window of at most one block is undivided in the on-duty band with
    ``f_on`` equal to the cycle share; 30 min at 1:1 is 15 + 15 with ``f_on``
    1; longer windows carry the exact cycle share in the phase mean with
    ``f_on`` 1.  Counts never fall below one exposure.

    Returns ``{band: {"f_on", "n_on_min", "n_on_mean", "n_on_max"}}``, the
    count statistics taken over the on-duty phases.
    """
    n_cycle = len(options)
    per_block = block_minutes * 60.0 / exptime_s
    n_blocks = n_all / per_block
    k = int(np.floor(n_blocks + 1e-9))
    rem = max(n_blocks - k, 0.0)
    out: dict[str, dict[str, float]] = {}
    for band in dict.fromkeys(options):
        counts = []
        for phase in range(n_cycle):
            n = sum(per_block for j in range(k) if options[(phase + j) % n_cycle] == band)
            if options[(phase + k) % n_cycle] == band:
                n += rem * per_block
            counts.append(n)
        on = [c for c in counts if c > 0.0]
        out[band] = {
            "f_on": len(on) / n_cycle,
            "n_on_min": max(min(on), 1.0) if on else 0.0,
            "n_on_mean": max(sum(on) / len(on), 1.0) if on else 0.0,
            "n_on_max": max(max(on), 1.0) if on else 0.0,
        }
    return out


def band_shares(options: list[str]) -> dict[str, float]:
    """Fraction of the filter-strategy cycle each band occupies.

    ``["g", "rho", "g", "rho"]`` gives 0.5 each; ``["g", "g", "rho"]`` gives
    2/3 and 1/3.  This is the long-run share of exposures a sky position
    receives in each band as it drifts across the alternating filter strips.
    """
    n = len(options)
    return {band: options.count(band) / n for band in dict.fromkeys(options)}


def _n_eff_for_cadence(
    window: str,
    season: str,
    f_clear: float,
    exptime_s: float,
    band_share: float = 1.0,
) -> float:
    """Effective number of coadded exposures for a cadence window.

    For windows up to one night, N_eff assumes the observer is on-sky
    (no weather loss).  For multi-night windows, ``f_clear`` scales
    out nights lost to weather and downtime.  ``band_share`` is the
    fraction of exposures taken in the band being evaluated (see
    :func:`band_shares`); it applies only to multi-exposure windows,
    and a single exposure is never split below one.

    Parameters
    ----------
    window : str
        One of :data:`DEFAULT_CADENCE_WINDOWS`.
    season : str
        Season key (``"winter"``, ``"equinox"``, ``"summer"``).
    f_clear : float
        Fraction of ratchets that are usable (weather + downtime).
        For multi-night windows this is the annual (or seasonal) mean
        clear fraction from the per-month weather draws.
    exptime_s : float
        Single-exposure integration time in seconds.
    band_share : float
        Fraction of the exposures in this window taken in the evaluated
        band (1.0 for a single-band strategy).  A window shorter than one
        exposure still yields one exposure, so N_eff is never below 1.

    """
    n_eff = _n_eff_all_bands(window, season, f_clear, exptime_s)
    if n_eff > 1.0:
        n_eff = n_eff * band_share
    return max(n_eff, 1.0)


def _cadence_n_eff_spec(window, season, f_clear, exptime_s, band_share, band, duty_model="block", options=None):
    """Share-mean N_eff plus the on-duty N_eff (p16/p50/p84 slots) and f_on for the band.

    ``duty_model="block"`` (default): :func:`block_duty` on the filter cycle
    ``options``; the three slots hold the phase minimum, phase mean and phase
    maximum of the on-duty count, so ``on_duty_p50`` is the phase-averaged
    block count and p16/p84 bracket its full spread over starting blocks.
    Single-band cycles return the share mean alone.
    """
    if duty_model not in DUTY_MODELS:
        raise ValueError(f"Unknown duty_model={duty_model!r}, expected one of {DUTY_MODELS}")
    share_mean = _n_eff_for_cadence(window, season, f_clear, exptime_s, band_share)
    if not options or len(dict.fromkeys(options)) < 2:
        return share_mean
    n_all = _n_eff_all_bands(window, season, f_clear, exptime_s)
    d = block_duty(list(options), n_all, exptime_s)[band]
    return {
        "share_mean": share_mean,
        "on_duty_p16": d["n_on_min"],
        "on_duty_p50": d["n_on_mean"],
        "on_duty_p84": d["n_on_max"],
        "f_on": d["f_on"],
    }


def _n_eff_all_bands(window: str, season: str, f_clear: float, exptime_s: float) -> float:
    exp_per_hour = 3600.0 / exptime_s
    night_h = _SEASON_NIGHT_HOURS[season]
    avg_night_h = sum(_SEASON_NIGHT_HOURS.values()) / len(_SEASON_NIGHT_HOURS)

    if window == "1sec":
        return 1.0
    elif window == "1min":
        return 60.0 / exptime_s
    elif window == "15min":
        return 15.0 * 60.0 / exptime_s
    elif window == "30min":
        return 30.0 * 60.0 / exptime_s
    elif window == "1hour":
        return exp_per_hour
    elif window == "1night":
        return night_h * exp_per_hour
    elif window == "1week":
        return 7.0 * night_h * exp_per_hour * f_clear
    elif window == "6month":
        return 182.5 * avg_night_h * exp_per_hour * f_clear
    elif window == "5year":
        return 5.0 * _NIGHTS_PER_YEAR * avg_night_h * exp_per_hour * f_clear
    else:
        raise ValueError(f"Unknown cadence window: {window}")


# ---------------------------------------------------------------------------
# Correlated per-ratchet draws
# ---------------------------------------------------------------------------


def draw_cloud_transparency(
    rng: np.random.Generator,
    photometric_fraction: float,
    median_mag: float,
    sigma_ln: float,
    max_mag: float,
) -> float:
    """Draw the grey transparency of one exposure on a clear night.

    The transparency is 1 with probability ``photometric_fraction``, otherwise 10^(-0.4 m) with the extra
    extinction m ~ lognormal(ln median_mag, sigma_ln) clipped at ``max_mag``.
    """
    if rng.random() < photometric_fraction:
        return 1.0
    extra = min(float(rng.lognormal(np.log(median_mag), sigma_ln)), max_mag)
    return float(10 ** (-0.4 * extra))


def draw_atmosphere(
    rng: np.random.Generator,
    seeing_mean: float,
    seeing_std: float,
    transparency_prior: Prior,
    rho: float = 0.5,
) -> tuple[float, float]:
    """Zenith seeing (lognormal) and grey transparency for one ratchet, from the transparency prior as stated.

    ``"normal"``: the latent-variable correlated draw (:func:`draw_correlated_atmosphere`) with the prior's own mean
    and std.  ``"cloud_lognormal"``: seeing lognormal, transparency from the cloud model, independent (no basis for
    correlating cirrus with seeing).
    """
    if transparency_prior.dist == "normal":
        return draw_correlated_atmosphere(
            rng, seeing_mean, seeing_std, transparency_prior.params["mean"], transparency_prior.params["std"], rho
        )
    seeing = float(np.exp(np.log(seeing_mean) + seeing_std * rng.normal(0, 1)))
    return seeing, transparency_prior.draw(rng)


def transparency_prior(priors) -> Prior:
    """Return the ``observatory.transparency`` entry of ``priors``, or that of DEFAULT_RATCHET_PRIORS."""
    return next(
        (p for p in priors if p.field == "observatory.transparency"),
        next(p for p in DEFAULT_RATCHET_PRIORS if p.field == "observatory.transparency"),
    )


def draw_correlated_atmosphere(
    rng: np.random.Generator,
    seeing_mean: float,
    seeing_std: float,
    transparency_mean: float,
    transparency_std: float,
    rho: float = 0.5,
) -> tuple[float, float]:
    """Draw correlated seeing and transparency from a latent quality variable.

    A latent atmospheric quality q ~ N(0,1) drives both seeing and
    transparency.  Worse atmosphere (larger q) produces worse seeing
    and lower transparency.

    The parameter ``rho`` controls coupling strength to the latent
    variable, but the resulting Pearson correlation between seeing and
    transparency z-scores is -rho**2, not -rho.  For example,
    rho=0.5 gives r = -0.25.

    """
    q = rng.normal(0, 1)
    noise_seeing = rng.normal(0, 1)
    noise_transp = rng.normal(0, 1)

    # Seeing: higher q → worse (larger) seeing
    z_seeing = rho * q + np.sqrt(1 - rho**2) * noise_seeing
    seeing = np.exp(np.log(seeing_mean) + seeing_std * z_seeing)

    # Transparency: higher q → worse (lower) transparency
    z_transp = -rho * q + np.sqrt(1 - rho**2) * noise_transp
    transparency = transparency_mean + transparency_std * z_transp
    transparency = float(np.clip(transparency, 0.5, 1.0))

    return float(seeing), transparency


# ---------------------------------------------------------------------------
# Single-image depth evaluation
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DepthResult:
    """Return type for evaluate_single_image_depth."""

    single_depths: dict[str, float]
    sat_mags: dict[str, float]
    psf_fwhm: dict[str, float]
    optical_psf_size: dict[str, float]
    cadence_depths: dict[str, dict[str, float]] = dataclasses.field(default_factory=dict)
    cadence_depths_on_duty: dict[str, dict[str, dict[str, float]]] = dataclasses.field(default_factory=dict)


def evaluate_single_image_depth(
    obs: Observatory,
    noise_budget: NoiseBudget,
    psf_model: GaussianPSF | DeliveredPSF,
    ab: ABPhot,
    band_params: dict,
    grid_point: RatchetGridPoint,
    seeing: float,
    transparency: float,
    pixel_phase: tuple[float, float] | None,
    cfg: Config,
    cadence_n_eff: dict[str, float] | None = None,
    elongation_arcsec: float = 0.0,
    elongation_angle_deg: float = 0.0,
) -> DepthResult:
    """Compute single-image limiting magnitude at one grid point.

    Optionally evaluates coadded depth at multiple cadence windows
    alongside the single-exposure depth.  Sky brightness, PSF, and
    throughput are computed once per band; the noise budget is
    re-evaluated for each ``coadd_n`` value, so the full CCD equation
    (including read-noise scaling) is used at every cadence.

    Parameters
    ----------
    obs : Observatory
        Observatory instance for sky brightness.
    noise_budget : NoiseBudget
        Noise budget calculator.
    psf_model : GaussianPSF or DeliveredPSF
        PSF model instance.
    ab : ABPhot
        AB magnitude system.
    band_params : dict
        Per-band throughput and sky electron rate parameters.
    grid_point : RatchetGridPoint
        Representative ratchet conditions.
    seeing : float
        Drawn atmospheric seeing (arcsec).
    transparency : float
        Drawn atmospheric transparency (0-1).
    pixel_phase : tuple[float, float] or None
        Subpixel centroid offset (x, y) in [0, 1).
    cfg : Config
        Survey configuration.
    cadence_n_eff : dict[str, float | dict[str, float]], optional
        Mapping of cadence window name to effective number of coadded
        exposures.  When provided, ``cadence_depths`` is populated.
    elongation_arcsec : float
        Tracking-induced PSF elongation in arcseconds.  Applied as a
        uniform line-kernel convolution after the PSF is generated.
        Default 0.0 (no elongation).
    elongation_angle_deg : float
        Position angle of the elongation trail in degrees, measured
        counterclockwise from the x-axis.  Default 0.0.

    Returns
    -------
    DepthResult
        Structured result with ``single_depths``, ``sat_mags``,
        ``psf_fwhm``, ``optical_psf_size``, and ``cadence_depths``
        (empty dict when ``cadence_n_eff`` is not provided).

    """
    sky_components = obs.sky_components_at(
        alt_deg=grid_point.alt_deg,
        az_deg=180.0,  # due south, arbitrary
        t=grid_point.time,
    )

    exptime = cfg.survey.base_cadence_s * u.second

    result = {}
    sat_mags: dict[str, float] = {}
    psf_fwhm: dict[str, float] = {}
    optical_psf_size: dict[str, float] = {}
    cadence_result: dict[str, dict[str, float]] = {w: {} for w in (cadence_n_eff or {})}
    cadence_on_duty: dict[str, dict[str, dict[str, float]]] = {w: {} for w in (cadence_n_eff or {})}

    for band, bp in band_params.items():
        # seeing is the zenith draw; the delivered seeing at this grid point's airmass is seeing * X**0.6
        psf_kwargs = {
            "seeing": float(seeing_at_airmass(seeing, grid_point.airmass)),
            "band": band,
            "return_optical_hfd": True,
        }
        if isinstance(psf_model, DeliveredPSF):
            if pixel_phase is not None:
                psf_kwargs["pixel_phase"] = pixel_phase
        psf_array, delivered_size, optical_size = psf_model.with_seeing(**psf_kwargs)
        if elongation_arcsec > 0:
            psf_array = elongate_psf(psf_array, elongation_arcsec, elongation_angle_deg, psf_model.plate_scale)
        psf_fwhm[band] = float(delivered_size)
        optical_psf_size[band] = float(optical_size)

        background_e = bp["sky"].sky_electrons(
            airglow_s10=sky_components["airglow_s10"],
            zodiacal_s10=sky_components["zodiacal_s10"],
            starlight_s10=sky_components["starlight_s10"],
            moon_s10=sky_components["moon_s10"],
            artif_s10=sky_components["artif_s10"],
            transparency=transparency,
        )
        background_e *= psf_model.plate_scale**2
        background_e = background_e * u.electron / u.second

        # Full-system signal throughput at this airmass (exact photon-rate integral, Beer-Lambert per wavelength),
        # then the grey transparency draw.
        effective_tp = float(signal_throughput(bp, grid_point.airmass)) * transparency

        signal_e, _ = noise_budget.get_flux_at_snr(
            snr=cfg.survey.detection_snr,
            psf=psf_array,
            eskyflux=background_e,
            exptime=exptime,
            signal_throughput=effective_tp,
            coadd_n=1,
        )

        signal_e /= exptime
        signal_photons = signal_e / bp["band_qe"]
        limmag = float(ab.mag_from_photons(band, signal_photons))
        result[band] = limmag

        peak_fraction = psf_array.max()
        fullwell = cfg.telescope.fullwell * u.electron
        bg_per_pix = background_e * exptime
        sat_flux_e = (fullwell - bg_per_pix) / (peak_fraction * exptime * effective_tp)
        sat_photons = sat_flux_e / bp["band_qe"]
        sat_mags[band] = float(ab.mag_from_photons(band, sat_photons))

        if cadence_n_eff:
            for window, n_eff_spec in cadence_n_eff.items():
                spec = n_eff_spec[band] if isinstance(n_eff_spec, dict) else n_eff_spec
                variants = spec if isinstance(spec, dict) else {"share_mean": spec}

                def _coadd_depth(n_eff):
                    signal_e_coadd, _ = noise_budget.get_flux_at_snr(
                        snr=cfg.survey.detection_snr,
                        psf=psf_array,
                        eskyflux=background_e,
                        exptime=exptime,
                        signal_throughput=effective_tp,
                        coadd_n=n_eff,
                    )
                    return float(ab.mag_from_photons(band, (signal_e_coadd / exptime) / bp["band_qe"]))

                cadence_result[window][band] = _coadd_depth(variants["share_mean"])
                if "on_duty_p50" in variants:
                    cadence_on_duty[window][band] = {
                        "p16": _coadd_depth(variants["on_duty_p16"]),
                        "p50": _coadd_depth(variants["on_duty_p50"]),
                        "p84": _coadd_depth(variants["on_duty_p84"]),
                        "f_on": float(variants["f_on"]),
                    }

    return DepthResult(
        single_depths=result,
        sat_mags=sat_mags,
        psf_fwhm=psf_fwhm,
        optical_psf_size=optical_psf_size,
        cadence_depths=cadence_result,
        cadence_depths_on_duty={w: v for w, v in cadence_on_duty.items() if v},
    )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SampleResult:
    """Summary statistics from a single forward MC sample."""

    sample_id: int
    theta: dict[str, float]
    ratchet_depths: dict[str, list[float]]
    median_depth: dict[str, float]
    depth_10pct: dict[str, float]
    depth_90pct: dict[str, float]
    cadence_depths: dict[str, dict[str, float]] = dataclasses.field(default_factory=dict)
    cadence_depths_on_duty: dict[str, dict[str, dict[str, float]]] = dataclasses.field(default_factory=dict)
    median_sat_mag: dict[str, float] = dataclasses.field(default_factory=dict)
    sat_mag_10pct: dict[str, float] = dataclasses.field(default_factory=dict)
    sat_mag_90pct: dict[str, float] = dataclasses.field(default_factory=dict)
    median_psf_fwhm: dict[str, float] = dataclasses.field(default_factory=dict)
    median_optical_psf: dict[str, float] = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# ForwardMC class
# ---------------------------------------------------------------------------


class ForwardMC:
    """Monte Carlo uncertainty propagation over instrument parameters.

    Estimates the uncertainty on the predicted depth by drawing
    instrument and site parameters (read noise, dark current, throughput,
    solar activity) from prior distributions fit to lab measurements and
    published site data, then evaluating depth at a fixed 36-point representative grid
    (3 seasons x 4 lunar phases x 3 airmasses) weighted by how often
    each condition occurs.

    Each MC sample perturbs the instrument model and produces a single
    weighted-median depth per band.  Running N samples builds a
    distribution of depth predictions, from which confidence intervals
    and parameter sensitivities are derived.

    This complements :class:`~argus_sim.survey.Survey`, which fixes the
    instrument model and simulates a real observing schedule to produce
    spatially-resolved depth maps.  Both share the same leaf-level depth
    calculation (:func:`evaluate_single_image_depth`), but Survey
    integrates over sky position while ForwardMC integrates over
    parameter space.

    Parameters
    ----------
    base_config : Config
        Baseline configuration to perturb.
    priors : list[Prior], optional
        Sample and ratchet priors in one list, split by field.  Used only when
        ``sample_priors`` is None.
    sample_priors : list[Prior], optional
        Per-sample parameter priors. Defaults to ``DEFAULT_SAMPLE_PRIORS``.
    ratchet_priors : list[Prior], optional
        Per-ratchet parameter priors. Defaults to ``DEFAULT_RATCHET_PRIORS``.
        Their ``observatory.transparency`` entry sets the transparency draw
        (:func:`draw_atmosphere`); the seeing follows the seasonal model, or
        ``observatory.seeing_mean`` and ``seeing_std`` when it is off.
    output_dir : str or Path
        Directory for MC results.
    seeing_transparency_rho : float
        Coupling of seeing and transparency when the transparency prior is
        ``"normal"`` (:func:`draw_correlated_atmosphere`).  Default 0.5.
    cadence_windows : tuple of str, optional
        Cadence windows to evaluate. Defaults to ``DEFAULT_CADENCE_WINDOWS``.
    lunar_mode : str
        One of ``LUNAR_MODES``; ``"bright"`` uses fast-mode (1-s) exposures.
    max_tracking_elongation : float
        Upper bound (arcsec) of the uniform per-ratchet tracking elongation
        draw.  Default 0 (none).
    duty_model : str
        One of ``DUTY_MODELS``; see :func:`block_duty`.

    """

    def __init__(
        self,
        base_config: Config,
        priors: list[Prior] | None = None,
        sample_priors: list[Prior] | None = None,
        ratchet_priors: list[Prior] | None = None,
        output_dir: str | Path = "output_mc",
        seeing_transparency_rho: float = 0.5,
        cadence_windows: tuple[str, ...] | None = None,
        lunar_mode: str = "base",
        max_tracking_elongation: float = 0.0,
        duty_model: str = "block",
    ) -> None:
        """Initialize the ForwardMC runner."""
        if lunar_mode not in LUNAR_MODES:
            raise ValueError(f"Unknown lunar_mode={lunar_mode!r}, expected one of {LUNAR_MODES}")
        if duty_model not in DUTY_MODELS:
            raise ValueError(f"Unknown duty_model={duty_model!r}, expected one of {DUTY_MODELS}")

        if priors is not None and sample_priors is None:
            self.priors = priors
            self._sample_priors = [
                p for p in priors if p.field not in {"observatory.seeing_mean", "observatory.transparency"}
            ]
            self._ratchet_priors = [
                p for p in priors if p.field in {"observatory.seeing_mean", "observatory.transparency"}
            ]
        else:
            self._sample_priors = sample_priors if sample_priors is not None else list(DEFAULT_SAMPLE_PRIORS)
            self._ratchet_priors = ratchet_priors if ratchet_priors is not None else list(DEFAULT_RATCHET_PRIORS)
            self.priors = self._sample_priors + self._ratchet_priors

        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self._seeing_transparency_rho = seeing_transparency_rho
        self.results: list[SampleResult] = []
        self._cadence_windows = cadence_windows if cadence_windows is not None else DEFAULT_CADENCE_WINDOWS
        self._lunar_mode = lunar_mode
        self._max_tracking_elongation = max_tracking_elongation
        self._duty_model = duty_model
        self._cradle = None
        self._grid: list[RatchetGridPoint] | None = None
        self._grid_weights: np.ndarray | None = None

    def _get_cradle(self):
        """Build CradlePointing once and cache it."""
        if self._cradle is None:
            from .grid import CradlePointing

            log.info("Building CradlePointing (one-time cost)...")
            self._cradle = CradlePointing(contract_lon=True)
            log.info("CradlePointing ready.")
        return self._cradle

    def _get_grid(self) -> tuple[list[RatchetGridPoint], np.ndarray]:
        """Build and cache the representative ratchet grid and weights."""
        if self._grid is None:
            self._grid = build_ratchet_grid(
                lunar_mode=self._lunar_mode,
                bright_threshold=self.base_config.survey.highspeed_start_moonfrac,
            )
            self._grid_weights = compute_grid_weights(self._grid)
        return self._grid, self._grid_weights

    def _run_single_sample(
        self,
        sample_id: int,
        rng: np.random.Generator,
    ) -> SampleResult:
        """Run one MC sample: perturb config, evaluate grid ratchets, scale."""
        cfg, theta = perturb_config(self.base_config, self._sample_priors, rng)

        if self._lunar_mode == "bright":
            cfg.survey.base_cadence_s = cfg.survey.fast_cadence_s

        obs = Observatory(config=cfg)
        tp_loss = cfg.telescope.throughput_loss
        throughputs = SystemThroughput(throughput_loss=tp_loss)
        ab = ABPhot(
            collecting_area=cfg.telescope.collecting_area,
            throughput_loss=tp_loss,
            throughput=throughputs,
        )
        noise_budget = NoiseBudget(
            read_noise=cfg.telescope.readnoise * u.electron,
            dark_current=cfg.telescope.dark_current * u.electron / u.second,
        )

        if cfg.telescope.psf_model == "delivered":
            psf_model = DeliveredPSF.from_config(cfg, randomize_pixel_phase=False)
        else:
            psf_model = GaussianPSF(
                stamp_size=cfg.survey.stamp_size,
                stamp_resolution=cfg.survey.stamp_resolution,
                plate_scale=cfg.telescope.plate_scale,
                pixel_size=cfg.telescope.pixel_size,
                rms_spot_size=cfg.telescope.spot_size,
            )

        band_params = build_band_params(sorted(set(cfg.filter_strategy.options)), throughputs, ab)
        shares = band_shares(list(cfg.filter_strategy.options))
        grid, weights = self._get_grid()

        use_seasonal = cfg.observatory.seasonal_seeing
        seeing_mean = cfg.observatory.seeing_mean
        seeing_std = cfg.observatory.seeing_std
        tr_prior = transparency_prior(self._ratchet_priors)

        ratchet_depths: dict[str, list[float]] = {}
        ratchet_sat_mags: dict[str, list[float]] = {}
        ratchet_psf_fwhm: dict[str, list[float]] = {}
        ratchet_optical_psf: dict[str, list[float]] = {}
        cadence_ratchet_depths: dict[str, dict[str, list[float]]] = {w: {} for w in self._cadence_windows}
        cadence_ratchet_on_duty: dict[str, dict[str, dict]] = {}

        exptime_s = cfg.survey.base_cadence_s

        monthly_weather = draw_monthly_weather(rng)
        f_clear = _annual_mean_clear_fraction(monthly_weather)
        for month, f in monthly_weather.items():
            theta[f"weather_month_{month:02d}"] = f

        for i, grid_pt in enumerate(grid):
            if use_seasonal:
                sm, ss = seasonal_seeing_params(grid_pt.time)
            else:
                sm, ss = seeing_mean, seeing_std

            seeing, transparency = draw_atmosphere(rng, sm, ss, tr_prior, rho=self._seeing_transparency_rho)

            pixel_phase: tuple[float, float] | None = None
            if isinstance(psf_model, DeliveredPSF):
                pixel_phase = (float(rng.uniform(0, 1)), float(rng.uniform(0, 1)))

            elongation = float(rng.uniform(0, self._max_tracking_elongation))
            elongation_angle = float(rng.uniform(0, 180))

            theta[f"ratchet_{i}.seeing"] = seeing
            theta[f"ratchet_{i}.transparency"] = transparency
            theta[f"ratchet_{i}.tracking_elongation"] = elongation
            theta[f"ratchet_{i}.tracking_angle"] = elongation_angle
            if pixel_phase is not None:
                theta[f"ratchet_{i}.pixel_phase_y"] = pixel_phase[0]
                theta[f"ratchet_{i}.pixel_phase_x"] = pixel_phase[1]

            cadence_n_eff = {
                w: {
                    band: _cadence_n_eff_spec(
                        w,
                        grid_pt.season,
                        f_clear,
                        exptime_s,
                        shares.get(band, 1.0),
                        band,
                        self._duty_model,
                        list(cfg.filter_strategy.options),
                    )
                    for band in band_params
                }
                for w in self._cadence_windows
            }

            dr = evaluate_single_image_depth(
                obs=obs,
                noise_budget=noise_budget,
                psf_model=psf_model,
                ab=ab,
                band_params=band_params,
                grid_point=grid_pt,
                seeing=seeing,
                transparency=transparency,
                pixel_phase=pixel_phase,
                cfg=cfg,
                cadence_n_eff=cadence_n_eff,
                elongation_arcsec=elongation,
                elongation_angle_deg=elongation_angle,
            )
            depths = dr.single_depths
            sat_mags_i = dr.sat_mags
            cadence_depths_i = dr.cadence_depths
            psf_sizes_i = dr.psf_fwhm
            optical_sizes_i = dr.optical_psf_size

            for band, size in psf_sizes_i.items():
                theta[f"ratchet_{i}.psf_fwhm_{band}"] = size
                ratchet_psf_fwhm.setdefault(band, []).append(size)
            for band, size in optical_sizes_i.items():
                theta[f"ratchet_{i}.optical_psf_{band}"] = size
                ratchet_optical_psf.setdefault(band, []).append(size)

            for band, mag in depths.items():
                ratchet_depths.setdefault(band, []).append(mag)

            for band, mag in sat_mags_i.items():
                ratchet_sat_mags.setdefault(band, []).append(mag)

            for window, band_mags in cadence_depths_i.items():
                for band, mag in band_mags.items():
                    cadence_ratchet_depths[window].setdefault(band, []).append(mag)
            for window, band_vals in dr.cadence_depths_on_duty.items():
                for band, vals in band_vals.items():
                    slot = cadence_ratchet_on_duty.setdefault(window, {}).setdefault(
                        band, {"p16": [], "p50": [], "p84": [], "f_on": vals["f_on"]}
                    )
                    for q in ("p16", "p50", "p84"):
                        slot[q].append(vals[q])

        median_depth: dict[str, float] = {}
        depth_10pct: dict[str, float] = {}
        depth_90pct: dict[str, float] = {}

        for band, mags in ratchet_depths.items():
            mags_arr = np.array(mags)
            median_depth[band] = float(_weighted_percentile(mags_arr, weights, 50))
            depth_10pct[band] = float(_weighted_percentile(mags_arr, weights, 10))
            depth_90pct[band] = float(_weighted_percentile(mags_arr, weights, 90))

        cadence_depths: dict[str, dict[str, float]] = {}
        for window in self._cadence_windows:
            cadence_depths[window] = {}
            for band, mags in cadence_ratchet_depths[window].items():
                mags_arr = np.array(mags)
                cadence_depths[window][band] = float(_weighted_percentile(mags_arr, weights, 50))

        cadence_depths_on_duty: dict[str, dict[str, dict[str, float]]] = {}
        for window, band_slots in cadence_ratchet_on_duty.items():
            cadence_depths_on_duty[window] = {}
            for band, slot in band_slots.items():
                cadence_depths_on_duty[window][band] = {
                    q: float(_weighted_percentile(np.array(slot[q]), weights, 50)) for q in ("p16", "p50", "p84")
                }
                cadence_depths_on_duty[window][band]["f_on"] = float(slot["f_on"])

        median_sat_mag: dict[str, float] = {}
        sat_mag_10pct: dict[str, float] = {}
        sat_mag_90pct: dict[str, float] = {}
        for band, mags in ratchet_sat_mags.items():
            mags_arr = np.array(mags)
            median_sat_mag[band] = float(_weighted_percentile(mags_arr, weights, 50))
            sat_mag_10pct[band] = float(_weighted_percentile(mags_arr, weights, 10))
            sat_mag_90pct[band] = float(_weighted_percentile(mags_arr, weights, 90))

        median_psf_fwhm: dict[str, float] = {}
        for band, fwhms in ratchet_psf_fwhm.items():
            fwhm_arr = np.array(fwhms)
            median_psf_fwhm[band] = float(_weighted_percentile(fwhm_arr, weights, 50))

        median_optical_psf: dict[str, float] = {}
        for band, sizes in ratchet_optical_psf.items():
            size_arr = np.array(sizes)
            median_optical_psf[band] = float(_weighted_percentile(size_arr, weights, 50))

        return SampleResult(
            sample_id=sample_id,
            theta=theta,
            ratchet_depths=ratchet_depths,
            median_depth=median_depth,
            depth_10pct=depth_10pct,
            depth_90pct=depth_90pct,
            cadence_depths=cadence_depths,
            cadence_depths_on_duty=cadence_depths_on_duty,
            median_sat_mag=median_sat_mag,
            sat_mag_10pct=sat_mag_10pct,
            sat_mag_90pct=sat_mag_90pct,
            median_psf_fwhm=median_psf_fwhm,
            median_optical_psf=median_optical_psf,
        )

    def run(
        self,
        n_samples: int,
        seed: int = 42,
        save_every: int = 10,
    ) -> list[SampleResult]:
        """Run the forward Monte Carlo ensemble.

        Parameters
        ----------
        n_samples : int
            Number of parameter draws.
        seed : int
            Master RNG seed for reproducibility.
        save_every : int
            Write intermediate results to disk every N samples.

        Returns
        -------
        list[SampleResult]
            Summary statistics for each sample.

        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)

        # Build the representative grid once, before the samples
        self._get_grid()

        for i in range(n_samples):
            log.info(f"Forward MC sample {i + 1}/{n_samples}")

            result = self._run_single_sample(i, rng)
            self.results.append(result)

            if (i + 1) % save_every == 0 or (i + 1) == n_samples:
                self._save_results()

            log.info(f"Sample {i + 1} complete: {result.median_depth}")

        return self.results

    def _save_results(self) -> None:
        """Write current results to JSON."""
        out = []
        for r in self.results:
            entry = {
                "sample_id": r.sample_id,
                "theta": {k: v for k, v in r.theta.items() if not k.startswith("ratchet_")},
                "median_depth": r.median_depth,
                "depth_10pct": r.depth_10pct,
                "depth_90pct": r.depth_90pct,
            }
            if r.cadence_depths:
                entry["cadence_depths"] = r.cadence_depths
            if r.cadence_depths_on_duty:
                entry["cadence_depths_on_duty"] = r.cadence_depths_on_duty
            if r.median_sat_mag:
                entry["median_sat_mag"] = r.median_sat_mag
            if r.sat_mag_10pct:
                entry["sat_mag_10pct"] = r.sat_mag_10pct
            if r.sat_mag_90pct:
                entry["sat_mag_90pct"] = r.sat_mag_90pct
            if r.median_psf_fwhm:
                entry["median_psf_fwhm"] = r.median_psf_fwhm
            if r.median_optical_psf:
                entry["median_optical_psf"] = r.median_optical_psf
            out.append(entry)
        path = self.output_dir / "mc_results.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        write_provenance(
            self.output_dir,
            "forward_mc",
            {
                "n_samples": len(out),
                "duty_model": self._duty_model,
                "filter_cycle": list(self.base_config.filter_strategy.options),
            },
        )
        log.info(f"Saved {len(out)} results to {path}")

    def to_dataframe(self):
        """Convert results to a pandas DataFrame for analysis."""
        import pandas as pd

        rows = []
        for r in self.results:
            row: dict[str, float | int] = {"sample_id": r.sample_id}
            for k, v in r.theta.items():
                if not k.startswith("ratchet_"):
                    row[f"theta_{k}"] = v
            for k, v in r.median_depth.items():
                row[f"median_depth_{k}"] = v
            for k, v in r.depth_10pct.items():
                row[f"depth_10pct_{k}"] = v
            for k, v in r.depth_90pct.items():
                row[f"depth_90pct_{k}"] = v
            for window, band_depths in r.cadence_depths.items():
                for band, mag in band_depths.items():
                    row[f"cadence_{window}_{band}"] = mag
            for band, mag in r.median_sat_mag.items():
                row[f"sat_mag_{band}"] = mag
            for band, mag in r.sat_mag_10pct.items():
                row[f"sat_mag_10pct_{band}"] = mag
            for band, mag in r.sat_mag_90pct.items():
                row[f"sat_mag_90pct_{band}"] = mag
            for band, fwhm in r.median_psf_fwhm.items():
                row[f"psf_fwhm_{band}"] = fwhm
            for band, size in r.median_optical_psf.items():
                row[f"optical_psf_{band}"] = size
            rows.append(row)
        return pd.DataFrame(rows)

    def sensitivity_table(self, depth_col: str | None = None):
        """Compute sensitivity of depth to each input parameter.

        For each input parameter, computes the Spearman rank correlation
        with the output depth and the depth range (16th–84th percentile)
        attributable to that parameter's variation.

        Parameters
        ----------
        depth_col : str, optional
            Name of the depth column to analyze (e.g. ``"median_depth_g"``).
            If None, uses the first ``median_depth_*`` column found.

        Returns
        -------
        pandas.DataFrame
            Table with columns: ``parameter``, ``spearman_rho``,
            ``p_value``, ``depth_range_16_84``.

        """
        df = self.to_dataframe()
        return compute_sensitivity(df, depth_col=depth_col)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_sensitivity(df, depth_col: str | None = None):
    """Compute Spearman rank correlations between input parameters and depth.

    For each ``theta_*`` column in the DataFrame, computes the Spearman
    rank correlation with the chosen depth column and the depth spread
    (16th–84th percentile) within the top and bottom quartiles of that
    parameter.

    Parameters
    ----------
    df : pandas.DataFrame
        Output from ``ForwardMC.to_dataframe()``. Must have ``theta_*``
        columns and at least one ``median_depth_*`` column.
    depth_col : str, optional
        Depth column to analyze. If None, uses the first
        ``median_depth_*`` column.

    Returns
    -------
    pandas.DataFrame
        Columns: ``parameter``, ``spearman_rho``, ``p_value``,
        ``depth_range_16_84``. Sorted by absolute correlation
        (most sensitive first).

    """
    import pandas as pd
    from scipy.stats import spearmanr

    theta_cols = [c for c in df.columns if c.startswith("theta_")]
    if not theta_cols:
        raise ValueError("No theta_* columns found in DataFrame")

    if depth_col is None:
        depth_cols = [c for c in df.columns if c.startswith("median_depth_")]
        if not depth_cols:
            raise ValueError("No median_depth_* columns found in DataFrame")
        depth_col = depth_cols[0]

    depth = df[depth_col].values
    rows = []
    for col in theta_cols:
        param_name = col.removeprefix("theta_")
        param_vals = df[col].values

        mask = np.isfinite(param_vals) & np.isfinite(depth)
        if mask.sum() < 3:
            continue

        rho, pval = spearmanr(param_vals[mask], depth[mask])

        # Depth spread: difference in median depth between the bottom
        # and top quartiles of this parameter.
        q25 = np.percentile(param_vals[mask], 25)
        q75 = np.percentile(param_vals[mask], 75)
        low_q = depth[mask & (param_vals <= q25)]
        high_q = depth[mask & (param_vals >= q75)]
        if len(low_q) > 0 and len(high_q) > 0:
            depth_range = float(np.median(high_q) - np.median(low_q))
        else:
            depth_range = np.nan

        rows.append(
            {
                "parameter": param_name,
                "spearman_rho": float(rho),
                "p_value": float(pval),
                "depth_range_16_84": depth_range,
            }
        )

    result = pd.DataFrame(rows)
    result = result.reindex(result["spearman_rho"].abs().sort_values(ascending=False).index)
    return result.reset_index(drop=True)


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    """Compute a weighted percentile.

    Parameters
    ----------
    values : np.ndarray
        Data values.
    weights : np.ndarray
        Non-negative weights (need not sum to 1).
    percentile : float
        Percentile in [0, 100].

    """
    sorted_idx = np.argsort(values)
    sorted_vals = values[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cum_weights = np.cumsum(sorted_weights)
    cum_weights /= cum_weights[-1]
    return float(np.interp(percentile / 100.0, cum_weights, sorted_vals))


# ---------------------------------------------------------------------------
# Prior visualization
# ---------------------------------------------------------------------------

# Human-readable labels and units for each prior field
_PRIOR_LABELS = {
    "telescope.readnoise": ("Read noise", "e⁻"),
    "telescope.dark_current": ("Dark current", "e⁻/s/pix"),
    "telescope.throughput_loss": ("Optics throughput", ""),
    "observatory.solar_activity": ("Solar activity", ""),
    "survey.fraction_of_ratchets_to_run": ("Clear fraction scale", ""),
    "observatory.seeing_mean": ("Seeing", "arcsec"),
    "observatory.transparency": ("Atm. transparency", ""),
}


def plot_priors(
    sample_priors: list[Prior] | None = None,
    ratchet_priors: list[Prior] | None = None,
    seeing_transparency_rho: float = 0.5,
    n_draws: int = 10000,
    seed: int = 42,
    save_path: str | None = None,
):
    """Visualize the MC prior distributions.

    Draws ``n_draws`` samples from each prior and plots histograms with
    summary statistics. Per-ratchet seeing and transparency are shown
    both as marginals and as a joint scatter colored by their latent
    atmospheric quality correlation.

    Parameters
    ----------
    sample_priors : list[Prior], optional
        Per-sample priors. Defaults to ``DEFAULT_SAMPLE_PRIORS``.
    ratchet_priors : list[Prior], optional
        Per-ratchet priors. Defaults to ``DEFAULT_RATCHET_PRIORS``.
    seeing_transparency_rho : float
        Correlation coefficient for correlated draws.
    n_draws : int
        Number of samples to draw for visualization.
    seed : int
        RNG seed.
    save_path : str, optional
        If provided, save figure to this path instead of showing.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.

    """
    import matplotlib.pyplot as plt

    if sample_priors is None:
        sample_priors = DEFAULT_SAMPLE_PRIORS
    if ratchet_priors is None:
        ratchet_priors = DEFAULT_RATCHET_PRIORS

    rng = np.random.default_rng(seed)

    # Draw from per-sample priors
    sample_draws = {}
    for prior in sample_priors:
        sample_draws[prior.field] = np.array([prior.draw(rng) for _ in range(n_draws)])

    # Draw correlated seeing + transparency
    seeing_prior = next((p for p in ratchet_priors if p.field == "observatory.seeing_mean"), None)
    transp_prior = next((p for p in ratchet_priors if p.field == "observatory.transparency"), None)

    seeing_draws = np.empty(n_draws)
    transp_draws = np.empty(n_draws)
    if seeing_prior is not None and transp_prior is not None:
        if seeing_prior.dist == "seasonal_lognormal":  # the marginals as the priors state them
            for i in range(n_draws):
                seeing_draws[i] = seeing_prior.draw(rng)
                transp_draws[i] = transp_prior.draw(rng)
        else:
            for i in range(n_draws):
                s, t = draw_atmosphere(
                    rng,
                    np.exp(seeing_prior.params["mean"])
                    if seeing_prior.dist == "lognormal"
                    else seeing_prior.params["mean"],
                    seeing_prior.params["std"],
                    transp_prior,
                    rho=seeing_transparency_rho,
                )
                seeing_draws[i] = s
                transp_draws[i] = t

    # Layout: per-sample priors + seeing marginal + transparency marginal
    # + joint scatter
    n_sample = len(sample_priors)
    n_panels = n_sample + 3  # seeing, transparency, joint
    n_cols = 3
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows))
    axes = axes.flatten()

    panel_idx = 0

    # Per-sample prior histograms
    for prior in sample_priors:
        ax = axes[panel_idx]
        draws = sample_draws[prior.field]
        label, unit = _PRIOR_LABELS.get(prior.field, (prior.field, ""))

        ax.hist(draws, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="none")
        ax.axvline(np.median(draws), color="k", ls="--", lw=1, label=f"median={np.median(draws):.4g}")
        ax.axvspan(
            np.percentile(draws, 16),
            np.percentile(draws, 84),
            alpha=0.15,
            color="k",
            label="16–84th pct",
        )

        xlabel = f"{label} [{unit}]" if unit else label
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(f"{label} ({prior.dist})")
        ax.legend(fontsize=7, loc="upper right")
        panel_idx += 1

    # Seeing marginal
    if seeing_prior is not None:
        ax = axes[panel_idx]
        ax.hist(seeing_draws, bins=50, density=True, alpha=0.7, color="coral", edgecolor="none")
        ax.axvline(np.median(seeing_draws), color="k", ls="--", lw=1, label=f"median={np.median(seeing_draws):.2f}")
        ax.axvspan(
            np.percentile(seeing_draws, 16),
            np.percentile(seeing_draws, 84),
            alpha=0.15,
            color="k",
            label="16–84th pct",
        )
        ax.set_xlabel("Seeing [arcsec]")
        ax.set_ylabel("Density")
        ax.set_title("Seeing (per-ratchet, lognormal)")
        ax.legend(fontsize=7, loc="upper right")
        panel_idx += 1

    # Transparency marginal
    if transp_prior is not None:
        ax = axes[panel_idx]
        ax.hist(transp_draws, bins=50, density=True, alpha=0.7, color="mediumseagreen", edgecolor="none")
        ax.axvline(np.median(transp_draws), color="k", ls="--", lw=1, label=f"median={np.median(transp_draws):.3f}")
        ax.axvspan(
            np.percentile(transp_draws, 16),
            np.percentile(transp_draws, 84),
            alpha=0.15,
            color="k",
            label="16–84th pct",
        )
        ax.set_xlabel("Atmospheric transparency")
        ax.set_ylabel("Density")
        ax.set_title("Transparency (per-ratchet, normal)")
        ax.legend(fontsize=7, loc="upper right")
        panel_idx += 1

    # Joint seeing–transparency scatter
    if seeing_prior is not None and transp_prior is not None:
        ax = axes[panel_idx]
        subsample = min(2000, n_draws)
        idx = rng.choice(n_draws, subsample, replace=False)
        sc = ax.scatter(
            seeing_draws[idx],
            transp_draws[idx],
            c=seeing_draws[idx],
            cmap="RdYlGn_r",
            s=4,
            alpha=0.5,
            rasterized=True,
        )
        ax.set_xlabel("Seeing [arcsec]")
        ax.set_ylabel("Transparency")
        measured_rho = np.corrcoef(seeing_draws, transp_draws)[0, 1]
        ax.set_title(f"Joint (ρ={measured_rho:.2f})")
        fig.colorbar(sc, ax=ax, label="Seeing [arcsec]", shrink=0.8)
        panel_idx += 1

    # Hide unused axes
    for i in range(panel_idx, len(axes)):
        axes[i].set_visible(False)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"Saved prior distributions to {save_path}")

    return fig


def plot_sensitivity(
    sensitivity_df=None,
    mc_df=None,
    depth_col: str | None = None,
    save_path: str | None = None,
):
    """Visualize the sensitivity table as a horizontal bar chart.

    Shows two panels side by side: Spearman rank correlation (left) and
    depth range between top/bottom quartiles (right), both sorted by
    absolute correlation strength.

    Parameters
    ----------
    sensitivity_df : pandas.DataFrame, optional
        Pre-computed sensitivity table from ``compute_sensitivity()``.
        If None, ``mc_df`` must be provided and the table is computed.
    mc_df : pandas.DataFrame, optional
        MC output DataFrame (from ``ForwardMC.to_dataframe()``). Used
        only if ``sensitivity_df`` is None.
    depth_col : str, optional
        Depth column to analyze. Passed to ``compute_sensitivity()``.
    save_path : str, optional
        If provided, save figure to this path instead of showing.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.

    """
    import matplotlib.pyplot as plt

    if sensitivity_df is None:
        if mc_df is None:
            raise ValueError("Provide either sensitivity_df or mc_df")
        sensitivity_df = compute_sensitivity(mc_df, depth_col=depth_col)

    st = sensitivity_df
    params = st["parameter"].values
    rhos = st["spearman_rho"].values
    depth_ranges = st["depth_range_16_84"].values

    # Human-readable labels
    labels = []
    for p in params:
        label, unit = _PRIOR_LABELS.get(p, (p.split(".")[-1], ""))
        labels.append(label)

    y = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, max(3, 0.5 * len(labels))))

    # Left panel: Spearman rho
    colors_rho = ["#c44e52" if r < 0 else "#4c72b0" for r in rhos]
    ax1.barh(y, rhos, color=colors_rho, edgecolor="none", height=0.6)
    ax1.axvline(0, color="k", lw=0.5)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("Spearman ρ")
    ax1.set_title("Rank correlation with depth")
    ax1.invert_yaxis()
    rho_pad = max(0.15, 0.25 * (np.abs(rhos).max() if len(rhos) > 0 else 0.5))
    ax1.set_xlim(min(rhos.min() - rho_pad, -0.1), max(rhos.max() + rho_pad, 0.1))

    # Annotate with p-value significance
    for i, (rho, pval) in enumerate(zip(rhos, st["p_value"].values)):
        if pval < 0.001:
            sig = "***"
        elif pval < 0.01:
            sig = "**"
        elif pval < 0.05:
            sig = "*"
        else:
            sig = ""
        # Place label on the outside end of the bar
        if rho >= 0:
            ax1.text(rho + 0.02, i, f"{rho:+.2f}{sig}", va="center", ha="left", fontsize=8)
        else:
            ax1.text(rho - 0.02, i, f"{rho:+.2f}{sig}", va="center", ha="right", fontsize=8)

    # Right panel: depth range
    colors_dr = ["#c44e52" if r < 0 else "#4c72b0" for r in depth_ranges]
    ax2.barh(y, depth_ranges, color=colors_dr, edgecolor="none", height=0.6)
    ax2.axvline(0, color="k", lw=0.5)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel("Δ depth [mag] (Q75 − Q25 of parameter)")
    ax2.set_title("Depth range by parameter quartile")
    ax2.invert_yaxis()
    finite_dr = depth_ranges[np.isfinite(depth_ranges)]
    if len(finite_dr) > 0:
        dr_pad = max(0.01, 0.3 * np.abs(finite_dr).max())
        ax2.set_xlim(min(finite_dr.min() - dr_pad, -0.005), max(finite_dr.max() + dr_pad, 0.005))

    for i, dr in enumerate(depth_ranges):
        if np.isfinite(dr):
            offset = 0.005 if dr >= 0 else -0.005
            ha = "left" if dr >= 0 else "right"
            ax2.text(dr + offset, i, f"{dr:+.3f}", va="center", ha=ha, fontsize=8)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"Saved sensitivity plot to {save_path}")

    return fig
