"""Closure of the helioecliptic longitude used by the zodiacal light model.

A target built through the Observatory's own AltAz path and pointed at the Sun
must have helioecliptic longitude 0; pointed at the antisun, 180.  The Sun's
longitude must therefore be taken in a geocentric ecliptic frame: in a
barycentric frame the Sun sits ~0.006 AU from the origin and its "longitude"
is the direction of the barycentre offset, not the apparent Sun.
"""

import astropy.coordinates as crds
import astropy.time as atime
import numpy as np
import pytest

from argus_sim.observatory import Observatory

DATES = [
    "2024-08-07T07:00:00",
    "2026-03-15T07:00:00",
    "2026-06-15T07:00:00",
    "2026-06-15T19:00:00",
    "2026-09-15T07:00:00",
    "2026-12-15T07:00:00",
]

TOL_DEG = 0.05


@pytest.fixture(scope="module")
def obs():
    return Observatory()


def _sun_altaz(obs, t):
    altaz = crds.AltAz(location=obs.el, obstime=t)
    sun = crds.get_body("sun", t).transform_to(altaz)
    return sun.alt.deg, sun.az.deg


@pytest.mark.parametrize("date", DATES)
def test_sun_pointed_target_has_zero_helioecliptic_longitude(obs, date):
    t = atime.Time(date)
    alt, az = _sun_altaz(obs, t)
    comp = obs.sky_components_at(alt, az, t)
    assert comp["helio_ecl_lon_deg"][0] == pytest.approx(0.0, abs=TOL_DEG)


@pytest.mark.parametrize("date", DATES)
def test_antisun_pointed_target_has_180_helioecliptic_longitude(obs, date):
    t = atime.Time(date)
    alt, az = _sun_altaz(obs, t)
    comp = obs.sky_components_at(-alt, (az + 180.0) % 360.0, t)
    assert comp["helio_ecl_lon_deg"][0] == pytest.approx(180.0, abs=TOL_DEG)


@pytest.mark.parametrize("date", DATES)
def test_helioecliptic_longitude_is_folded(obs, date):
    t = atime.Time(date)
    alt = np.array([30.0, 60.0, 89.0])
    az = np.array([0.0, 120.0, 240.0])
    comp = obs.sky_components_at(alt, az, t)
    assert np.all(comp["helio_ecl_lon_deg"] >= 0.0)
    assert np.all(comp["helio_ecl_lon_deg"] <= 180.0)
    assert np.all(comp["abs_beta_deg"] <= 90.0)
