"""Tests for arbitrary source-SED integration through ArgusSim."""

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest
import scipy.integrate as sing

from argus_sim import ABPhot, SystemThroughput, c
from argus_sim.spectral_sky import build_band_params, signal_throughput


@pytest.fixture(scope="module")
def setup_photometry():
    """Share the Argus throughput grid and collecting area across SED tests."""
    tp = SystemThroughput(throughput_loss=1.0, filters={"beta": True, "rho": True})
    ab = ABPhot(throughput=tp, collecting_area=c.telescope.collecting_area)
    return tp, ab


@pytest.mark.parametrize("band", ["beta", "rho"])
def test_flat_fnu_sed_matches_existing_flux_density_path(setup_photometry, band):
    """A flat F_nu SED must reproduce photons_from_flux_density."""
    tp, ab = setup_photometry
    fnu = 2.3e-27 * u.erg / (u.second * u.cm**2 * u.Hz)
    sed_fnu = np.full(tp.wav.size, fnu.value) * fnu.unit

    got = ab.photons_from_sed(band, tp.wav, sed_fnu)
    expected = ab.photons_from_flux_density(band, fnu)

    np.testing.assert_allclose(got.to_value(u.photon / u.second), expected.to_value(u.photon / u.second), rtol=2e-4)


@pytest.mark.parametrize("band", ["beta", "rho"])
def test_flambda_and_fnu_inputs_are_equivalent(setup_photometry, band):
    """Equivalent F_lambda and F_nu representations must give the same result."""
    tp, ab = setup_photometry
    fnu = (1 + (tp.wav.value / 500) ** 2) * 1.7e-4 * u.Jy
    flam = fnu.to(u.erg / (u.second * u.cm**2 * u.AA), equivalencies=u.spectral_density(tp.wav))

    from_fnu = ab.electrons_from_sed(band, tp.wav, fnu, airmass=1.3)
    from_flam = ab.electrons_from_sed(band, tp.wav, flam, airmass=1.3)

    np.testing.assert_allclose(from_fnu.value, from_flam.value, rtol=1e-12)
    np.testing.assert_allclose(
        ab.photons_from_sed(band, tp.wav, fnu).value,
        ab.photons_from_sed(band, tp.wav.to(u.AA), flam).value,
        rtol=1e-12,
    )


@pytest.mark.parametrize("band", ["beta", "rho"])
@pytest.mark.parametrize("mag", [14.0, 22.0])
def test_sed_recovers_known_ab_magnitude(setup_photometry, band, mag):
    """Both magnitude paths recover the flat-F_nu input AB magnitude."""
    tp, ab = setup_photometry
    fnu = np.ones(tp.wav.size) * ab.zp_ergs * 10 ** (-0.4 * mag)
    photons = ab.photons_from_sed(band, tp.wav, fnu)
    assert float(ab.mag_from_photons(band, photons)) == pytest.approx(mag, abs=1e-6)
    assert float(ab.mag_from_sed(band, tp.wav, fnu)) == pytest.approx(mag, abs=1e-6)


@pytest.mark.parametrize("band", ["beta", "rho"])
@pytest.mark.parametrize("airmass", [0.0, 1.0, 1.37, 2.0])
def test_flat_fnu_full_system_matches_existing_ab_treatment(setup_photometry, band, airmass):
    """Direct SED integration must agree with the survey's exact AB-source throughput."""
    tp, ab = setup_photometry
    fnu = 3.1e-27 * u.erg / (u.second * u.cm**2 * u.Hz)
    sed_fnu = np.full(tp.wav.size, fnu.value) * fnu.unit
    bp = build_band_params([band], tp, ab)[band]

    got = ab.electrons_from_sed(band, tp.wav, sed_fnu, airmass=airmass)
    filter_photons = ab.photons_from_flux_density(band, fnu)
    expected = filter_photons * bp["band_qe"] * signal_throughput(bp, airmass)

    np.testing.assert_allclose(got.to_value(u.electron / u.second), expected.to_value(u.electron / u.second), rtol=3e-4)


def test_transparency_is_grey_multiplicative_factor(setup_photometry):
    """Grey transparency scales the detected electron rate linearly."""
    tp, ab = setup_photometry
    fnu = np.full(tp.wav.size, 2.0e-27) * u.erg / (u.second * u.cm**2 * u.Hz)
    clear = ab.electrons_from_sed("beta", tp.wav, fnu, airmass=1.2, transparency=1.0)
    grey = ab.electrons_from_sed("beta", tp.wav, fnu, airmass=1.2, transparency=0.73)
    np.testing.assert_allclose(grey.value, 0.73 * clear.value, rtol=1e-12)


def test_partial_band_coverage_is_rejected_by_default(setup_photometry):
    """An incomplete SED requires explicit permission for partial coverage."""
    _, ab = setup_photometry
    wav = np.linspace(400, 450, 100) * u.nm
    flux = np.ones(wav.size) * 1e-15 * u.erg / (u.second * u.cm**2 * u.nm)

    with pytest.raises(ValueError, match="does not cover the full"):
        ab.photons_from_sed("beta", wav, flux)

    rate = ab.photons_from_sed("beta", wav, flux, allow_partial=True)
    assert rate > 0 * u.photon / u.second


@pytest.mark.parametrize("band", ["beta", "rho"])
def test_sampled_sed_interpolation_and_partial_integration(setup_photometry, band):
    """Integrate a sloped SED on its overlap without extrapolating its flux."""
    tp, ab = setup_photometry
    wav = np.array([425, 450, 600, 650]) * u.nm
    flux = (wav.value / 500) * 1e-15 * u.erg / (u.s * u.cm**2 * u.nm)
    values = np.where((tp.wav >= wav[0]) & (tp.wav <= wav[-1]), tp.wav.value / 500 * 1e-15, 0)
    energy = (const.h * const.c / tp.wav).to_value(u.erg)
    expected = sing.trapezoid(values * tp.filt_wav[band] / energy, tp.wav.value)
    expected *= ab.collecting_area.to_value(u.cm**2)
    got = ab.photons_from_sed(band, wav[::-1], flux[::-1], allow_partial=True)
    assert got.to_value(u.photon / u.s) == pytest.approx(expected, rel=1e-12)
    assert np.isfinite(ab.mag_from_sed(band, wav, flux, allow_partial=True))
    assert ab.electrons_from_sed(band, wav, flux, allow_partial=True) > 0 * u.electron / u.s
    for method in (ab.photons_from_sed, ab.mag_from_sed, ab.electrons_from_sed):
        with pytest.raises(ValueError, match="does not cover the full"):
            method(band, wav, flux)


@pytest.mark.parametrize(
    "wav,flux,error,match",
    [
        ([300, 1000], np.ones(2) * u.Jy, TypeError, "Quantity"),
        (np.array([300, 1000]) * u.nm, [1, 1], TypeError, "Quantity"),
        (np.array([[300, 1000]]) * u.nm, np.ones((1, 2)) * u.Jy, ValueError, "one-dimensional"),
        (np.array([300, 1000]) * u.nm, np.ones(3) * u.Jy, ValueError, "same shape"),
        (np.array([300]) * u.nm, np.ones(1) * u.Jy, ValueError, "at least two"),
        (np.array([300, 1000]) * u.Hz, np.ones(2) * u.Jy, u.UnitConversionError, "wavelength units"),
        (np.array([300, 1000]) * u.nm, np.ones(2) * u.erg, u.UnitConversionError, "spectral flux"),
        (np.array([300, 300, 1000]) * u.nm, np.ones(3) * u.Jy, ValueError, "unique"),
        (np.array([0, 1000]) * u.nm, np.ones(2) * u.Jy, ValueError, "positive"),
        (np.array([-300, 1000]) * u.nm, np.ones(2) * u.Jy, ValueError, "positive"),
        (np.array([np.nan, 1000]) * u.nm, np.ones(2) * u.Jy, ValueError, "finite"),
        (np.array([300, np.inf]) * u.nm, np.ones(2) * u.Jy, ValueError, "finite"),
        (np.array([300, 1000]) * u.nm, np.array([np.nan, 1]) * u.Jy, ValueError, "finite"),
        (np.array([300, 1000]) * u.nm, np.array([1, np.inf]) * u.Jy, ValueError, "finite"),
    ],
)
def test_invalid_sed_inputs(setup_photometry, wav, flux, error, match):
    """Invalid arrays and units fail before integration."""
    _, ab = setup_photometry
    with pytest.raises(error, match=match):
        ab.photons_from_sed("beta", wav, flux)


def test_disjoint_partial_sed_has_zero_rate(setup_photometry):
    """Partial coverage does not extrapolate a disjoint SED into the band."""
    _, ab = setup_photometry
    wav = np.array([1100, 1200]) * u.nm
    flux = np.ones(2) * u.Jy
    assert ab.photons_from_sed("beta", wav, flux, allow_partial=True).value == 0
    assert ab.electrons_from_sed("beta", wav, flux, allow_partial=True).value == 0
