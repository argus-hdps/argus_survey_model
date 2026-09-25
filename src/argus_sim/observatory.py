"""Observatory location, observing conditions, and sky brightness models."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import astroplan
import astropy.coordinates as crds
import astropy.time as atime
import astropy.units as u
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from . import c
from .data.zodiacal_leinert1998 import ABS_BETA_DEG, ELONGATION_DEG, ZODIACAL_S10_FILLED
from .ab_system import ABPhot
from .spectral_sky import SpectralSky
from .throughput import SystemThroughput

if TYPE_CHECKING:
    from .config import Config


SEEING_AIRMASS_EXPONENT = 0.6  # Kolmogorov: FWHM scales as sec(z)^(3/5) (e.g. Roddier 1981)


def seeing_at_airmass(seeing_zenith, airmass):
    """Seeing FWHM at ``airmass`` from the zenith value: seeing_zenith * X**0.6."""
    return (
        np.asarray(seeing_zenith, dtype=np.float64) * np.asarray(airmass, dtype=np.float64) ** SEEING_AIRMASS_EXPONENT
    )


def pickering_airmass(alt_deg: np.ndarray | float) -> np.ndarray | float:
    """Airmass via Pickering (2002, DIO 12(2), Eq. 3).

    Stays finite at the horizon, where sec(z) diverges.
    """
    return 1.0 / np.sin(np.radians(alt_deg + 244.0 / (165.0 + 47.0 * np.power(alt_deg, 1.1))))


# Seasonal seeing from a two-DIMM campaign (Barker et al. 2003, Proc. SPIE 4837, 225-236, 2003SPIE.4837..225B,
# DOI 10.1117/12.458215).  Nightly medians (FWHM), corrected to the zenith and to a 10 ms integration, over 186 nights
# between July 2001 and July 2002:
#
# Winter (Dec-Feb) median: 1.24 +/- 0.33 arcsec
# Spring/Summer/Fall (Mar-Nov) median: 0.93 +/- 0.18 arcsec
# Seeing < 0.70 arcsec ~9% of the time.
#
# Two caveats from the paper: the DIMMs sat at ground level, so these include a ground layer the telescope may not
# see ("not quite lower limits" to the site performance); and the +/- values are the arcsec spread of the nightly
# medians, which draw_seasonal_seeing uses as the sigma of a lognormal (a wider spread: 0.42 arcsec in winter).
#
# The model uses these numbers as given (0.93 / 1.24 by season, annual median 0.97 arcsec over 3 winter months in 12)
# with the lognormal reading of the spread.
site_seeing = {
    1: {"median": 1.24, "std": 0.33},  # winter
    2: {"median": 1.24, "std": 0.33},
    3: {"median": 0.93, "std": 0.18},  # spring
    4: {"median": 0.93, "std": 0.18},
    5: {"median": 0.93, "std": 0.18},
    6: {"median": 0.93, "std": 0.18},  # summer
    7: {"median": 0.93, "std": 0.18},
    8: {"median": 0.93, "std": 0.18},
    9: {"median": 0.93, "std": 0.18},  # fall
    10: {"median": 0.93, "std": 0.18},
    11: {"median": 0.93, "std": 0.18},
    12: {"median": 1.24, "std": 0.33},  # winter
}


def seasonal_seeing_params(epoch):
    """Return (median, std) seeing in arcsec for the given epoch's season.

    Uses the DIMM seasons of Barker et al. (2003, 2003SPIE.4837..225B; see :data:`site_seeing`).

    Parameters
    ----------
    epoch : astropy.time.Time
        The observation epoch.

    Returns
    -------
    tuple[float, float]
        (median_seeing_arcsec, std_arcsec) for the season.

    """
    month = epoch.datetime.month
    params = site_seeing[month]
    return params["median"], params["std"]


# Clear-night fractions by month and ~10-day interval, from 19 years
# (2005-2024) of Clear Sky Chart archival data for a representative site (cleardarksky.com), which
# derives cloud-cover predictions from the Canadian Meteorological Centre's
# numerical weather model. Cross-validated against OpenWeatherMap hourly
# cloud-cover records; the two sources agree within 3 percentage points in
# all months. Annual-mean clear fraction ~68%, ranging from ~55% in late
# summer (Jul-Aug convective storms) to ~80% in winter (Dec-Feb).
site_weather = {
    1: {
        1: 0.75,
        11: 0.78,
        21: 0.63,
    },
    2: {
        1: 0.71,
        11: 0.67,
        21: 0.73,
    },
    3: {
        1: 0.66,
        11: 0.58,
        21: 0.71,
    },
    4: {
        1: 0.69,
        11: 0.70,
        21: 0.78,
    },
    5: {
        1: 0.79,
        11: 0.71,
        21: 0.74,
    },
    6: {
        1: 0.67,
        11: 0.63,
        21: 0.63,
    },
    7: {
        1: 0.58,
        11: 0.64,
        21: 0.61,
    },
    8: {
        1: 0.66,
        11: 0.59,
        21: 0.57,
    },
    9: {
        1: 0.59,
        11: 0.53,
        21: 0.60,
    },
    10: {1: 0.67, 11: 0.67, 21: 0.78},
    11: {
        1: 0.76,
        11: 0.74,
        21: 0.71,
    },
    12: {1: 0.67, 11: 0.68, 21: 0.80},
}


def observable_weather(epoch, rng=None):
    """Draw whether the night of ``epoch`` is clear.

    One Bernoulli draw with the clear fraction of ``site_weather`` for the
    epoch's month and the most recent ~10-day period.

    Parameters
    ----------
    epoch : astropy.time.Time
        The epoch to check.
    rng : numpy.random.Generator, optional
        Random number generator. If None, creates a fresh non-reproducible one.

    Returns
    -------
    bool
        True if the weather is observable, False if not.

    """
    if rng is None:
        rng = np.random.default_rng()

    month = epoch.datetime.month
    day = epoch.datetime.day

    month_data = site_weather[month]
    k = np.array(sorted(month_data.keys()))
    day_ix = k[day >= k][-1]
    good_weather_prob = month_data[day_ix]

    return bool(rng.random() < good_weather_prob)


class LocalEphem:
    """A class to handle local ephemerides calculations, including sunrise, sunset, moonrise, and moonset times.

    Methods
    -------
    sunset(day)
        Get the sunset time for the specified day.
    sunrise(day)
        Get the sunrise time for the specified day.
    moonset(day)
        Get the moonset time for the specified day.
    moonrise(day)
        Get the moonrise time for the specified day.
    moon_pos(obstime)
        Get the position of the moon at the specified observation time.
    moon_illum(obstime)
        Get the illumination of the moon at the specified observation time.

    """

    def __init__(self, location: crds.EarthLocation) -> None:
        """Initialize the LocalEphem object.

        Parameters
        ----------
        location : crds.EarthLocation
            The location of the observer.

        Returns
        -------
        None

        """
        self.observer = astroplan.Observer(location=location, timezone=c.observatory.timezone)

    def sunset(self, day: atime.Time) -> atime.Time:
        """Get the sunset time for the specified day.

        Parameters
        ----------
        day : atime.Time
            The day for which to get the sunset time.

        Returns
        -------
        atime.Time
            The sunset time.

        """
        return self.observer.sun_set_time(day, horizon=c.survey.min_sun_alt * u.deg, which="previous")

    def sunrise(self, day: atime.Time) -> atime.Time:
        """Get the sunrise time for the specified day.

        Parameters
        ----------
        day : atime.Time
            The day for which to get the sunrise time.

        Returns
        -------
        atime.Time
            The sunrise time.

        """
        return self.observer.sun_rise_time(day, horizon=c.survey.min_sun_alt * u.deg, which="next")

    def moonset(self, day: atime.Time) -> atime.Time:
        """Get the moonset time for the specified day.

        Parameters
        ----------
        day : atime.Time
            The day for which to get the moonset time.

        Returns
        -------
        atime.Time
            The moonset time.

        """
        return self.observer.moon_set_time(day)

    def moonrise(self, day: atime.Time) -> atime.Time:
        """Get the moonrise time for the specified day.

        Parameters
        ----------
        day : atime.Time
            The day for which to get the moonrise time.

        Returns
        -------
        atime.Time
            The moonrise time.

        """
        return self.observer.moon_rise_time(day)

    def moon_pos(self, obstime: atime.Time) -> crds.SkyCoord:
        """Get the position of the moon at the specified observation time.

        Parameters
        ----------
        obstime : atime.Time
            The observation time.

        Returns
        -------
        astropy.coordinates.SkyCoord
            The position of the moon.

        """
        return crds.get_body("moon", obstime)

    def moon_illum(self, obstime: atime.Time) -> float:
        """Get the illumination of the moon at the specified observation time.

        Parameters
        ----------
        obstime : atime.Time
            The observation time.

        Returns
        -------
        float
            The illumination of the moon.

        """
        return astroplan.moon_illumination(obstime)


class Observatory:
    """A class to handle observatory-related calculations, including sky brightness and weather verification.

    Methods
    -------
    owm_weather_verification(weather)
        Check an OpenWeatherMap-style weather record against observing limits.
    sky_brightness_at(alt_deg, az_deg, t, band="V", spaceman=False)
        Calculate the sky brightness at a given altitude, azimuth, and time.

    """

    def __init__(self, config: Config | None = None) -> None:
        """Initialize the Observatory object.

        Parameters
        ----------
        config : Config, optional
            Configuration object. Falls back to the global ``c`` singleton
            when not provided.

        """
        cfg = config if config is not None else c
        self.latitude = cfg.observatory.latitude * u.deg
        self.longitude = cfg.observatory.longitude * u.deg
        self.altitude = cfg.observatory.altitude * u.meter

        self.el = crds.EarthLocation(
            lat=self.latitude,
            lon=self.longitude,
            height=self.altitude,
        )

        self.artificial_light_cd_m2 = 1e-6 * cfg.observatory.artificial_light_cd_m2
        self._spectral_sky_cache = {}
        self._config = cfg

        self._zodiacal_interp = RegularGridInterpolator(
            (ELONGATION_DEG, ABS_BETA_DEG),
            ZODIACAL_S10_FILLED,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def _get_spectral_sky(self, band: str) -> SpectralSky:
        """Lazily construct and cache a SpectralSky instance for a band."""
        if band not in self._spectral_sky_cache:
            if not hasattr(self, "_throughput"):
                self._throughput = SystemThroughput(throughput_loss=1.0)
                self._ab = ABPhot(
                    collecting_area=self._config.telescope.collecting_area,
                    throughput_loss=1.0,
                    throughput=self._throughput,
                )
            self._spectral_sky_cache[band] = SpectralSky(self._throughput, self._ab, band)
        return self._spectral_sky_cache[band]

    @staticmethod
    def owm_weather_verification(weather: dict) -> tuple[bool, str]:
        """Check an OpenWeatherMap-style weather record against observing limits (cloud cover, dew point, ...).

        Parameters
        ----------
        weather : dict
            The weather record (``clouds_all``, ``temp``, ``dew_point``, ...).

        Returns
        -------
        tuple
            A tuple containing a boolean indicating whether the weather conditions are acceptable and a string describing the reason if not.

        """
        if weather["clouds_all"] >= 30:
            return False, "cloud"
        if (weather["temp"] + 3.5) < weather["dew_point"]:
            return False, "dew"
        if weather["rain_3h"] > 0 or weather["snow_3h"] > 0:
            return False, "precip"
        if weather["wind_speed"] > 20:
            return False, "wind"
        if weather["humidity"] > 80:
            return False, "humidity"
        return True, "none"

    def sky_components_at(
        self,
        alt_deg: float,
        az_deg: float,
        t: atime.Time,
        spaceman: bool = False,
    ) -> dict:
        """Calculate individual sky brightness components in S_10 units.

        Returns a dict of S_10 values for each sky component (dark, moon,
        artificial) plus geometric metadata, for use with spectral sky models.

        Parameters
        ----------
        alt_deg : float
            The altitude in degrees.
        az_deg : float
            The azimuth in degrees.
        t : atime.Time
            The observation time.
        spaceman : bool
            If True, compute space-based sky (no airglow, moon, or artificial).

        Returns
        -------
        dict
            Keys: dark_s10, airglow_s10, zodiacal_s10, starlight_s10,
                  moon_s10, artif_s10, v_mag,
                  helio_ecl_lon_deg, abs_beta_deg,
                  moon_distance_deg, moon_alt, sun_alt, illum_frac

        """
        altaz = crds.AltAz(location=self.el, obstime=t)

        if not isinstance(alt_deg, np.ndarray):
            alt_deg = np.array([alt_deg])
        if not isinstance(az_deg, np.ndarray):
            az_deg = np.array([az_deg])

        target = crds.SkyCoord(alt=alt_deg * u.deg, az=az_deg * u.deg, frame=altaz)
        moon = crds.get_body("moon", t)
        sun = crds.get_body("sun", t)

        elongation = moon.separation(sun)
        moon_phase = np.degrees(
            np.arctan2(
                sun.distance * np.sin(np.deg2rad(elongation.deg)),
                moon.distance - sun.distance * np.cos(np.deg2rad(elongation.deg)),
            ),
        ).value

        illum_frac = (1 + np.cos(moon_phase * np.pi / 180)) / 2

        moon_altaz = moon.transform_to(altaz)
        moon_distance = moon_altaz.separation(target)
        sun_alt = sun.transform_to(altaz).alt.deg
        moon_zenith_distance = 90.0 - moon_altaz.alt.deg
        moon_alt = moon_altaz.alt.deg

        galactic = target.galactic
        galactic_latitude = galactic.b

        ecliptic = target.barycentrictrueecliptic
        ecliptic_latitude = ecliptic.lat

        # Helioecliptic longitude: angular distance from the Sun in ecliptic
        # longitude, folded to 0-180 deg (symmetric about the Sun-antisun axis).
        sun_ecl_lon = crds.get_body("sun", t).geocentrictrueecliptic.lon
        helio_ecl_lon = np.abs((ecliptic.lon - sun_ecl_lon).wrap_at(180 * u.deg).deg)

        airmass = pickering_airmass(alt_deg)

        # Dark-sky component models follow the ING Sky Brightness guide
        # (Benn & Ellison 1998) which consolidates Walker (1987) Table 1.
        # Solar activity index S_sun ranges 0.8 (min) to 2.0 (max).
        s_sun = self._config.observatory.solar_activity
        # V-band extinction coefficient k_V = 0.172 mag/airmass at a
        # good site (Krisciunas & Schaefer 1991, Table 2).
        extinction = 0.172

        # Airglow: Walker (1987), Eq. 1; scales with solar activity
        # and airmass (van Rhijn factor ~ sec(z) for a thin emitting layer).
        airglow = (145.0 + 130.0 * (s_sun - 0.8) / 1.2) * airmass

        # Zodiacal light: Leinert et al. (1998, A&AS 127, 1), Table 17:
        # 2D interpolation over (helioecliptic longitude, |ecliptic latitude|).
        abs_beta = np.abs(ecliptic_latitude.deg)
        zodiacal = self._zodiacal_interp(np.column_stack([helio_ecl_lon, abs_beta]))

        # Integrated starlight: Roach & Gordon (1973), Table 3:
        # exponential fall-off with galactic latitude, scale height ~10 deg.
        starlight = 100.0 * np.exp(-np.fabs(galactic_latitude.deg) / 10.0)

        # Lunar sky brightness model: Krisciunas & Schaefer (1991, PASP 103, 1033).
        # I*: illuminance of the Moon outside the atmosphere (K&S Eq. 9).
        moon_intensity = np.power(10.0, -0.4 * (3.84 + 0.026 * math.fabs(moon_phase) + 4e-9 * moon_phase**4))
        # f(rho): scattering function vs angular distance rho (K&S Eq. 16,
        # Rayleigh + Mie terms).
        f_p = 10**5.36 * (1.06 + np.cos(moon_distance.radian) ** 2) + np.power(10.0, 6.15 - moon_distance.deg / 40.0)
        # X(z): airmass approximation for a scattering atmosphere (K&S Eq. 3).
        x_z = (1.0 - 0.96 * np.sin(np.radians(90.0 - alt_deg)) ** 2) ** -0.5
        x_zm = (1.0 - 0.96 * np.sin(np.radians(moon_zenith_distance)) ** 2) ** -0.5
        # B_moon: lunar contribution in nanoLamberts (K&S Eq. 20).
        b_moon_nLambert = (  # noqa: N806
            f_p
            * moon_intensity
            * np.power(10.0, -0.4 * extinction * x_zm)
            * (1.0 - np.power(10.0, -0.4 * extinction * x_z))
        )

        # Unit conversion: 21.587 V mag/arcsec^2 = 79.0 nL (K&S 1991, Eq. 1)
        # and that surface brightness equals 300.05 S_10 (tenth-mag stars
        # per square degree).
        b_moon_s10 = (b_moon_nLambert / 79.0) * 300.05
        # K&S 1991 holds for a Moon above the horizon; its X(Z_moon) is symmetric about the horizon, so a set Moon
        # would otherwise light the sky as much as a risen one.
        b_moon_s10 = np.where(np.asarray(moon_alt) > 0.0, b_moon_s10, 0.0)

        # Conversion: 1 nL = 10^-5 / pi cd/m^2 (definition of the Lambert).
        artificial_light_nLambert = self.artificial_light_cd_m2 / (1e-5 / math.pi)  # noqa: N806
        artificial_light_s10 = (artificial_light_nLambert / 79.0) * 300.05

        # Sum for V-band magnitude (spectral color corrections are handled
        # per-component in SpectralSky.sky_electrons).
        dark_s10 = airglow + zodiacal + starlight
        moon_s10 = b_moon_s10
        artif_s10 = artificial_light_s10

        if spaceman:
            airglow_s10 = np.zeros_like(zodiacal)
            dark_s10 = zodiacal + starlight
            moon_s10 = np.zeros_like(dark_s10)
            artif_s10 = np.zeros_like(dark_s10)
            total_s10 = dark_s10
        else:
            airglow_s10 = airglow
            total_s10 = dark_s10 + moon_s10 + artif_s10

        # 1 S_10 = V 27.78 mag/arcsec^2 (Allen 1973, Astrophysical Quantities).
        v_mag = 27.78 - 2.5 * np.log10(total_s10)

        return {
            "dark_s10": dark_s10,
            "airglow_s10": airglow_s10,
            "zodiacal_s10": zodiacal,
            "starlight_s10": starlight,
            "moon_s10": moon_s10,
            "artif_s10": artif_s10,
            "v_mag": v_mag,
            "helio_ecl_lon_deg": helio_ecl_lon,
            "abs_beta_deg": abs_beta,
            "moon_distance_deg": moon_distance.deg,
            "moon_alt": moon_alt,
            "sun_alt": sun_alt,
            "illum_frac": illum_frac,
        }

    def sky_brightness_at(
        self,
        alt_deg: float,
        az_deg: float,
        t: atime.Time,
        band: str = "V",
        spaceman: bool = False,
    ) -> tuple[float, float, float, float, float]:
        """Calculate the sky brightness at a given altitude, azimuth, and time.

        Adapted from a script by NML, with a low Earth orbit option and vectorized calculations.

        Parameters
        ----------
        alt_deg : float
            The altitude in degrees.
        az_deg : float
            The azimuth in degrees.
        t : atime.Time
            The observation time.
        band : str
            Filter for sky brightness measurement. Default is 'V'; any filter SystemThroughput loads.
        spaceman : bool
            If True, compute sky brightness for a low Earth orbit observer. Default is False.

        Returns
        -------
        tuple
            Sky brightness (mag per square arcsecond in ``band``), Moon distance (deg), Moon altitude (deg),
            Sun altitude (deg) and lunar illumination fraction.

        """
        comp = self.sky_components_at(alt_deg, az_deg, t, spaceman=spaceman)
        v_mag = comp["v_mag"]

        if band != "V":
            sky_v = self._get_spectral_sky("V")
            sky_band = self._get_spectral_sky(band)

            v_electrons = sky_v.sky_electrons(
                comp["airglow_s10"],
                comp["zodiacal_s10"],
                comp["starlight_s10"],
                comp["moon_s10"],
                comp["artif_s10"],
            )
            band_electrons = sky_band.sky_electrons(
                comp["airglow_s10"],
                comp["zodiacal_s10"],
                comp["starlight_s10"],
                comp["moon_s10"],
                comp["artif_s10"],
            )

            v_mag = v_mag - 2.5 * np.log10(band_electrons / v_electrons)

        return v_mag, comp["moon_distance_deg"], comp["moon_alt"], comp["sun_alt"], comp["illum_frac"]
