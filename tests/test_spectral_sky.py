"""Tests for the spectral sky background model.

Covers:
  1. V-band consistency: spectral model vs the flat-spectrum conversion (1 S_10 = V 27.78)
  2. Color direction: moon vs airglow coefficient by band
  3. Per-component spectral shape: airglow OH bands dominate i-band
  4. sky_brightness_at agrees with sky_components_at
  5. sky_components_at round-trip: components sum to total V mag
  6. Coefficient positivity and finiteness
  7. Template column validation at load time
"""

import numpy as np
import pytest
import astropy.time as atime

from argus_sim import SystemThroughput, ABPhot, SpectralSky
from argus_sim.observatory import Observatory
from argus_sim.spectral_sky import _load_template


# --- Fixtures ---


@pytest.fixture(scope="module")
def collecting_area():
    """Standard Argus collecting area."""
    from argus_sim import c

    return c.telescope.collecting_area


@pytest.fixture(scope="module")
def throughput_all():
    """SystemThroughput with all filters."""
    return SystemThroughput(throughput_loss=1.0)


@pytest.fixture(scope="module")
def ab(collecting_area, throughput_all):
    """ABPhot instance."""
    return ABPhot(collecting_area=collecting_area, throughput_loss=1.0, throughput=throughput_all)


@pytest.fixture(scope="module")
def obs():
    """Observatory instance."""
    return Observatory()


@pytest.fixture(scope="module")
def dark_time():
    """A dark-time epoch (new moon, midnight)."""
    return atime.Time("2025-03-29T06:00:00", scale="utc")


@pytest.fixture(scope="module")
def bright_time():
    """A bright-time epoch (full moon, midnight)."""
    return atime.Time("2025-01-15T03:00:00", scale="utc")


# --- Tests ---


class TestVBandConsistency:
    """The V-band spectral model should closely match the flat-spectrum conversion."""

    @pytest.mark.parametrize("component", ["airglow", "zodiacal", "starlight"])
    def test_v_band_component_matches_flat_spectrum(self, throughput_all, ab, component):
        """Each V-band component coefficient should match the flat-spectrum
        path to within 5%, since V-band normalization makes color correction ~1.
        """
        sky_v = SpectralSky(throughput_all, ab, "V")
        coeff = getattr(sky_v, f"e_per_s10_{component}")

        # Flat-spectrum conversion: 1 S_10 = V mag 27.78
        flat_photons = ab.photons_from_mag("V", 27.78)
        sky_tp = throughput_all.optics["V"]
        band_qe = np.average(throughput_all.qe_freq, weights=throughput_all.filt_freq["V"])
        flat_e_per_s10 = flat_photons.value * sky_tp * band_qe.value

        ratio = coeff / flat_e_per_s10
        assert 0.95 < ratio < 1.05, f"V-band {component} ratio spectral/flat = {ratio:.4f}, expected ~1.0"

    def test_v_band_moon_coefficient_near_airglow(self, throughput_all, ab):
        """In V-band, moon and airglow coefficients should be similar (both
        V-normalized), differing only by spectral shape within the V filter.
        """
        sky_v = SpectralSky(throughput_all, ab, "V")
        ratio = sky_v.e_per_s10_moon / sky_v.e_per_s10_airglow
        assert 0.90 < ratio < 1.10, f"V-band moon/airglow ratio = {ratio:.4f}, expected ~1.0"


class TestColorDirection:
    """Spectral color corrections should go in the physically expected direction."""

    def test_g_band_moon_greater_than_airglow(self, throughput_all, ab):
        """In g-band, scattered moonlight (Rayleigh-boosted blue) should
        produce more electrons per S_10 than the airglow component.
        """
        sky_g = SpectralSky(throughput_all, ab, "g")
        assert sky_g.e_per_s10_moon > sky_g.e_per_s10_airglow, (
            f"g-band: moon ({sky_g.e_per_s10_moon:.4e}) should exceed airglow ({sky_g.e_per_s10_airglow:.4e})"
        )

    def test_i_band_moon_airglow_ratio_less_than_g(self, throughput_all, ab):
        """In i-band, the moon/airglow ratio should be smaller than in g-band
        (Rayleigh scattering blue-boosts moonlight, airglow is OH-dominated in red).
        """
        sky_g = SpectralSky(throughput_all, ab, "g")
        sky_i = SpectralSky(throughput_all, ab, "i")
        ratio_g = sky_g.e_per_s10_moon / sky_g.e_per_s10_airglow
        ratio_i = sky_i.e_per_s10_moon / sky_i.e_per_s10_airglow
        assert ratio_g > ratio_i, f"g-band moon/airglow ratio ({ratio_g:.4f}) should exceed i-band ({ratio_i:.4f})"


class TestPerComponentCoefficients:
    """Per-component spectral templates should produce physically distinct coefficients."""

    def test_airglow_i_band_exceeds_zodiacal(self, throughput_all, ab):
        """Airglow i-band coefficient should exceed zodiacal due to OH Meinel bands."""
        sky_i = SpectralSky(throughput_all, ab, "i")
        assert sky_i.e_per_s10_airglow > sky_i.e_per_s10_zodiacal, (
            f"i-band: airglow ({sky_i.e_per_s10_airglow:.4e}) should exceed "
            f"zodiacal ({sky_i.e_per_s10_zodiacal:.4e}) due to OH emission"
        )

    @pytest.mark.parametrize("band", ["g", "r", "i"])
    def test_per_component_coefficients_positive_and_finite(self, throughput_all, ab, band):
        sky = SpectralSky(throughput_all, ab, band)
        for name in ("e_per_s10_airglow", "e_per_s10_zodiacal", "e_per_s10_starlight"):
            val = getattr(sky, name)
            assert np.isfinite(val), f"{band} {name} is not finite: {val}"
            assert val > 0, f"{band} {name} is not positive: {val}"

    @pytest.mark.parametrize("band", ["g", "r", "i"])
    def test_components_are_spectrally_distinct(self, throughput_all, ab, band):
        """The three dark-sky components should not all have the same coefficient."""
        sky = SpectralSky(throughput_all, ab, band)
        coeffs = [sky.e_per_s10_airglow, sky.e_per_s10_zodiacal, sky.e_per_s10_starlight]
        assert len(set(f"{c:.6e}" for c in coeffs)) > 1, f"{band}: all three component coefficients are identical"


class TestCoefficientSanity:
    """Basic sanity checks on precomputed coefficients."""

    @pytest.mark.parametrize("band", ["g", "r", "i"])
    def test_coefficients_positive_and_finite(self, throughput_all, ab, band):
        sky = SpectralSky(throughput_all, ab, band)
        for name in (
            "e_per_s10_airglow",
            "e_per_s10_zodiacal",
            "e_per_s10_starlight",
            "e_per_s10_moon",
            "e_per_s10_artif",
        ):
            val = getattr(sky, name)
            assert np.isfinite(val), f"{band} {name} is not finite: {val}"
            assert val > 0, f"{band} {name} is not positive: {val}"

    @pytest.mark.parametrize(
        "band,s10",
        [
            ("g", 50),
            ("g", 200),
            ("r", 100),
            ("r", 500),
            ("i", 150),
            ("i", 1000),
        ],
    )
    def test_sky_electrons_linearity(self, throughput_all, ab, band, s10):
        """sky_electrons should scale linearly with S_10 values."""
        sky = SpectralSky(throughput_all, ab, band)
        e1 = sky.sky_electrons(s10, 0, 0, 0, 0)
        e2 = sky.sky_electrons(2 * s10, 0, 0, 0, 0)
        assert np.isclose(e2, 2 * e1, rtol=1e-10), f"Linearity violated for {band} at S_10={s10}"

    @pytest.mark.parametrize(
        "band,airglow,zodiacal,starlight,moon,artif",
        [
            ("g", 50, 30, 20, 50, 10),
            ("g", 200, 60, 40, 0, 0),
            ("r", 40, 20, 20, 200, 5),
            ("r", 0, 0, 0, 150, 30),
            ("i", 120, 80, 50, 100, 20),
            ("i", 0, 0, 0, 0, 500),
        ],
    )
    def test_sky_electrons_additivity(self, throughput_all, ab, band, airglow, zodiacal, starlight, moon, artif):
        """sky_electrons with all components should equal sum of individual contributions."""
        sky = SpectralSky(throughput_all, ab, band)
        e_total = sky.sky_electrons(airglow, zodiacal, starlight, moon, artif)
        e_parts = (
            sky.sky_electrons(airglow, 0, 0, 0, 0)
            + sky.sky_electrons(0, zodiacal, 0, 0, 0)
            + sky.sky_electrons(0, 0, starlight, 0, 0)
            + sky.sky_electrons(0, 0, 0, moon, 0)
            + sky.sky_electrons(0, 0, 0, 0, artif)
        )
        assert np.isclose(e_total, e_parts, rtol=1e-10)


class TestSkyBrightnessConsistency:
    """sky_brightness_at agrees with the component model it is built on."""

    def test_sky_brightness_returns_five_tuple(self, obs, dark_time):
        result = obs.sky_brightness_at(45.0, 90.0, dark_time)
        assert len(result) == 5

    def test_sky_brightness_matches_components_vmag(self, obs, dark_time):
        """sky_brightness_at in V-band should equal sky_components_at v_mag."""
        v_mag, *_ = obs.sky_brightness_at(45.0, 90.0, dark_time, band="V")
        comp = obs.sky_components_at(45.0, 90.0, dark_time)
        assert np.allclose(v_mag, comp["v_mag"])

    def test_sky_brightness_band_correction(self, obs, bright_time):
        """sky_brightness_at in g-band should differ from V-band by the band color."""
        v_mag, *_ = obs.sky_brightness_at(45.0, 90.0, bright_time, band="V")
        g_mag, *_ = obs.sky_brightness_at(45.0, 90.0, bright_time, band="g")
        # g should be brighter (lower mag number) than V for typical sky
        assert not np.allclose(v_mag, g_mag), "g and V should differ"


class TestComponentsRoundTrip:
    """sky_components_at S_10 values should reconstruct the V magnitude."""

    OBSERVING_TIMES = [
        atime.Time("2025-03-29T06:00:00", scale="utc"),  # new moon
        atime.Time("2025-01-15T03:00:00", scale="utc"),  # full moon
        atime.Time("2025-06-20T04:00:00", scale="utc"),  # summer
        atime.Time("2025-10-10T02:00:00", scale="utc"),  # fall
        atime.Time("2025-12-25T05:00:00", scale="utc"),  # winter
    ]

    @pytest.mark.parametrize(
        "alt,az,time_idx",
        [
            (30.0, 90.0, 0),
            (45.0, 180.0, 1),
            (60.0, 270.0, 2),
            (75.0, 45.0, 3),
            (90.0, 0.0, 4),
        ],
    )
    def test_components_sum_to_vmag(self, obs, alt, az, time_idx):
        t = self.OBSERVING_TIMES[time_idx]
        comp = obs.sky_components_at(alt, az, t)
        total_s10 = comp["dark_s10"] + comp["moon_s10"] + comp["artif_s10"]
        expected_vmag = 27.78 - 2.5 * np.log10(total_s10)
        assert np.allclose(comp["v_mag"], expected_vmag, atol=1e-8)

    @pytest.mark.parametrize("alt", [30.0, 45.0, 60.0, 75.0, 90.0])
    def test_dark_time_moon_negligible(self, obs, dark_time, alt):
        """During new moon, moon_s10 should be small compared to dark_s10."""
        comp = obs.sky_components_at(alt, 90.0, dark_time)
        assert comp["moon_s10"].mean() < comp["dark_s10"].mean()

    @pytest.mark.parametrize("alt", [30.0, 45.0, 60.0, 75.0, 90.0])
    def test_components_all_positive(self, obs, dark_time, alt):
        comp = obs.sky_components_at(alt, 180.0, dark_time)
        assert np.all(comp["dark_s10"] > 0)
        assert np.all(comp["moon_s10"] >= 0)
        assert np.all(comp["artif_s10"] >= 0)


class TestTemplateColumnValidation:
    """_load_template must reject files with wrong column names."""

    def test_missing_flux_column_raises(self, tmp_path):
        bad_file = tmp_path / "bad_template.csv"
        bad_file.write_text("wav,intensity\n3000.0,1.0\n4000.0,2.0\n")
        wav_nm = np.linspace(300, 1000, 100)
        with pytest.raises(ValueError, match="missing required columns.*flux"):
            _load_template(str(bad_file), wav_nm)

    def test_missing_wav_column_raises(self, tmp_path):
        bad_file = tmp_path / "bad_template.csv"
        bad_file.write_text("wavelength,flux\n3000.0,1.0\n4000.0,2.0\n")
        wav_nm = np.linspace(300, 1000, 100)
        with pytest.raises(ValueError, match="missing required columns.*wav"):
            _load_template(str(bad_file), wav_nm)

    def test_both_columns_missing_raises(self, tmp_path):
        bad_file = tmp_path / "bad_template.csv"
        bad_file.write_text("wavelength,intensity\n3000.0,1.0\n4000.0,2.0\n")
        wav_nm = np.linspace(300, 1000, 100)
        with pytest.raises(ValueError, match="missing required columns"):
            _load_template(str(bad_file), wav_nm)

    def test_valid_columns_accepted(self, tmp_path):
        good_file = tmp_path / "good_template.csv"
        good_file.write_text("wav,flux\n3000.0,1.0\n4000.0,2.0\n")
        wav_nm = np.linspace(300, 1000, 100)
        result = _load_template(str(good_file), wav_nm)
        assert result.shape == wav_nm.shape
