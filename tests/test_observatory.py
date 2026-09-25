import numpy as np
import pytest

import astropy.coordinates as crds
import astropy.time as atime
import astropy.units as u

from argus_sim import ABPhot, SpectralSky, SystemThroughput, c
from argus_sim.observatory import LocalEphem, Observatory


@pytest.fixture(scope="module")
def obs():
    return Observatory()


@pytest.fixture(scope="module")
def throughput():
    return SystemThroughput(throughput_loss=1.0)


@pytest.fixture(scope="module")
def ab(throughput):
    collecting_area = c.telescope.collecting_area
    return ABPhot(collecting_area=collecting_area, throughput_loss=1.0, throughput=throughput)


def test_sky_brightness_g_brighter_than_v(obs):
    """Under a moonlit sky (scattered sunlight, blue) the g sky is brighter than V."""
    t = atime.Time("2022-01-14T06:00:00", scale="utc")  # Moon at 64 deg altitude, 88% illuminated
    v_mag, *_ = obs.sky_brightness_at(90, 90, t, band="V")
    g_mag, *_ = obs.sky_brightness_at(90, 90, t, band="g")

    assert g_mag < v_mag


@pytest.mark.parametrize("band", ["g", "r", "i"])
def test_sky_brightness_consistent_with_spectral_sky(obs, throughput, ab, band):
    """sky_brightness_at band corrections should match SpectralSky component-weighted colors."""
    t = atime.Time("2025-01-15T03:00:00", scale="utc")

    v_mag, *_ = obs.sky_brightness_at(45.0, 90.0, t, band="V")
    band_mag, *_ = obs.sky_brightness_at(45.0, 90.0, t, band=band)

    comp = obs.sky_components_at(45.0, 90.0, t)
    sky_v = SpectralSky(throughput, ab, "V")
    sky_band = SpectralSky(throughput, ab, band)

    v_e = sky_v.sky_electrons(
        comp["airglow_s10"],
        comp["zodiacal_s10"],
        comp["starlight_s10"],
        comp["moon_s10"],
        comp["artif_s10"],
    )
    band_e = sky_band.sky_electrons(
        comp["airglow_s10"],
        comp["zodiacal_s10"],
        comp["starlight_s10"],
        comp["moon_s10"],
        comp["artif_s10"],
    )

    expected_band_mag = v_mag - 2.5 * np.log10(band_e / v_e)
    np.testing.assert_allclose(band_mag, expected_band_mag, atol=1e-6)


class TestWeatherLookup:
    """observable_weather must select the most recent 10-day period for the given day."""

    @pytest.mark.parametrize(
        "month,day,expected_period",
        [
            (1, 1, 1),
            (1, 5, 1),
            (1, 10, 1),
            (1, 11, 11),
            (1, 15, 11),
            (1, 20, 11),
            (1, 21, 21),
            (1, 25, 21),
            (1, 31, 21),
            (6, 1, 1),
            (6, 11, 11),
            (6, 21, 21),
            (12, 28, 21),
        ],
    )
    def test_weather_period_selection(self, month, day, expected_period):
        """observable_weather frequency should match the probability for the correct period."""
        from argus_sim.observatory import site_weather, observable_weather

        expected_prob = site_weather[month][expected_period]
        epoch = atime.Time(f"2025-{month:02d}-{day:02d}T12:00:00", scale="utc")

        rng = np.random.default_rng(42)
        n_trials = 5000
        results = [observable_weather(epoch, rng=rng) for _ in range(n_trials)]
        observed_freq = sum(results) / n_trials

        assert observed_freq == pytest.approx(expected_prob, abs=0.03), (
            f"month={month}, day={day}: observed freq {observed_freq:.3f}, "
            f"expected {expected_prob} (period {expected_period})"
        )


def test_moonset_before_moonrise():
    """Moonset should return when the moon sets, moonrise when it rises.

    On 2025-02-05 (first quarter) at the survey site the moon sets
    around 01:30 CST and rises around 12:16 CST.  Querying from local
    midnight the nearest moonset should precede the nearest moonrise.
    """
    location = crds.EarthLocation(
        lat=c.observatory.latitude * u.deg,
        lon=c.observatory.longitude * u.deg,
        height=c.observatory.altitude * u.m,
    )
    ephem = LocalEphem(location)
    midnight_utc = atime.Time("2025-02-05T06:00:00", scale="utc")

    moonset_time = ephem.moonset(midnight_utc)
    moonrise_time = ephem.moonrise(midnight_utc)

    assert moonset_time < moonrise_time


if __name__ == "__main__":
    pytest.main()


def test_a_set_moon_adds_no_sky_light(obs):
    """Moonlight is zero with the Moon below the horizon and positive with it well above and partly lit."""
    times = atime.Time(62500.0 + np.arange(120) * 0.25, format="mjd")
    set_seen = risen_seen = 0
    for t in times:
        comp = obs.sky_components_at(60.0, 180.0, t)
        moon_s10 = float(np.ravel(comp["moon_s10"])[0])
        moon_alt = float(np.ravel(comp["moon_alt"])[0])
        if moon_alt <= 0.0:
            assert moon_s10 == 0.0
            set_seen += 1
        elif moon_alt > 5.0 and float(np.ravel(comp["illum_frac"])[0]) > 0.2:
            assert moon_s10 > 0.0
            risen_seen += 1
    assert set_seen > 10 and risen_seen > 10
