"""Property tests for NoiseBudget SNR calculations.

Covers:
  1. Background-limited SNR scales with sharpness = sum(psf^2)
  2. get_snr / get_flux_at_snr round-trip inverse consistency
  3. signal_throughput round-trip: recovered * tp equals detected electrons
  4. End-to-end limiting magnitude self-consistency across ABPhot, NoiseBudget, SpectralSky
"""

import numpy as np
import astropy.units as u
import pytest
from hypothesis import given, settings, assume
from hypothesis.strategies import floats, sampled_from

from argus_sim.photon_budget import NoiseBudget
from argus_sim.throughput import SystemThroughput
from argus_sim.ab_system import ABPhot
from argus_sim.spectral_sky import SpectralSky
from argus_sim import c as config


def _make_gaussian_psf(sigma_pix):
    """Return a sum-normalized 2D Gaussian PSF on a pixel grid."""
    size = max(int(8 * sigma_pix), 7)
    if size % 2 == 0:
        size += 1
    center = size // 2
    y, x = np.mgrid[:size, :size]
    psf = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma_pix**2))
    psf /= psf.sum()
    return psf


@pytest.mark.parametrize("sigma_pix", [0.8, 1.5, 2.5, 4.0, 6.0])
def test_background_limited_snr_scales_with_sharpness(sigma_pix):
    """Background-limited SNR = signal * sqrt(sum(psf^2)) / sqrt(sky).

    For a sum-normalized PSF, sharpness S = sum(psf^2). The effective number
    of background noise pixels is 1/S (STScI WFPC2 Handbook Eq 6.5), giving:
        noise^2 ≈ sky_electrons / S
        SNR ≈ signal_electrons * sqrt(S) / sqrt(sky_electrons)
    """
    psf = _make_gaussian_psf(sigma_pix)
    sharpness_correct = np.sum(psf**2)

    nb = NoiseBudget(
        read_noise=0.0 * u.electron,
        dark_current=0.0 * u.electron / u.second,
    )
    eflux = 1.0 * u.electron / u.second
    eskyflux = 1e6 * u.electron / u.second
    exptime = 1.0 * u.second

    snr, _ = nb.get_snr(eflux, psf, eskyflux, exptime)

    signal_e = (eflux * exptime).value
    sky_e = (eskyflux * exptime).value
    expected_snr = signal_e * np.sqrt(sharpness_correct) / np.sqrt(sky_e)

    np.testing.assert_allclose(
        snr,
        expected_snr,
        rtol=1e-3,
        err_msg=f"sigma={sigma_pix}: SNR={snr:.6f}, expected={expected_snr:.6f}",
    )


@pytest.mark.parametrize(
    "sigma_pix,eflux_val,eskyflux_val,exptime_val",
    [
        (1.0, 100, 500, 10),
        (2.0, 50, 1000, 30),
        (1.5, 200, 200, 5),
        (3.0, 10, 5000, 60),
        (0.8, 500, 100, 1),
    ],
)
def test_snr_flux_round_trip_fixed(sigma_pix, eflux_val, eskyflux_val, exptime_val):
    """Round-trip with hand-picked parameter combinations."""
    _assert_snr_flux_round_trip(sigma_pix, eflux_val, eskyflux_val, exptime_val)


@given(
    sigma_pix=floats(0.5, 8.0),
    eflux_val=floats(0.1, 1e4),
    eskyflux_val=floats(1.0, 1e5),
    exptime_val=floats(0.1, 300.0),
)
@settings(max_examples=200)
def test_snr_flux_round_trip(sigma_pix, eflux_val, eskyflux_val, exptime_val):
    """get_flux_at_snr(get_snr(flux, ...)) should recover the original flux.

    For any valid (flux, sky, psf, exptime), computing SNR via get_snr then
    inverting via get_flux_at_snr with signal_throughput=1 and coadd_n=1 should
    recover the original signal electrons within tolerance.
    """
    _assert_snr_flux_round_trip(sigma_pix, eflux_val, eskyflux_val, exptime_val)


def _assert_snr_flux_round_trip(sigma_pix, eflux_val, eskyflux_val, exptime_val):
    psf = _make_gaussian_psf(sigma_pix)
    nb = NoiseBudget()

    eflux = eflux_val * u.electron / u.second
    eskyflux = eskyflux_val * u.electron / u.second
    exptime = exptime_val * u.second

    snr, _ = nb.get_snr(eflux, psf, eskyflux, exptime)

    recovered_electrons, _ = nb.get_flux_at_snr(
        snr=snr,
        psf=psf,
        eskyflux=eskyflux,
        exptime=exptime,
        signal_throughput=1.0,
        coadd_n=1,
    )

    original_electrons = (eflux * exptime).value

    np.testing.assert_allclose(
        recovered_electrons.value,
        original_electrons,
        rtol=1e-6,
        err_msg=(f"Round-trip failed: original={original_electrons:.2f}, recovered={recovered_electrons.value:.2f}"),
    )


@pytest.mark.parametrize(
    "sigma_pix,eflux_val,eskyflux_val,exptime_val",
    [
        (1.0, 100, 500, 10),
        (2.0, 50, 1000, 30),
        (1.5, 200, 200, 5),
        (3.0, 10, 5000, 60),
        (0.8, 500, 100, 1),
    ],
)
def test_snr_flux_round_trip_with_signal_throughput_fixed(sigma_pix, eflux_val, eskyflux_val, exptime_val):
    """Round-trip with throughput, hand-picked parameter combinations."""
    _assert_snr_flux_round_trip_with_throughput(sigma_pix, eflux_val, eskyflux_val, exptime_val, tp=0.5)


@given(
    sigma_pix=floats(0.5, 8.0),
    eflux_val=floats(0.1, 1e4),
    eskyflux_val=floats(1.0, 1e5),
    exptime_val=floats(0.1, 300.0),
    tp=floats(0.01, 1.0),
)
@settings(max_examples=200)
def test_snr_flux_round_trip_with_signal_throughput(sigma_pix, eflux_val, eskyflux_val, exptime_val, tp):
    """get_flux_at_snr returns above-atmosphere equivalent when signal_throughput < 1.

    For any signal_throughput, the recovered value represents the above-atmosphere
    flux. Multiplying by the throughput must recover the original detected
    electrons.
    """
    _assert_snr_flux_round_trip_with_throughput(sigma_pix, eflux_val, eskyflux_val, exptime_val, tp)


def _assert_snr_flux_round_trip_with_throughput(sigma_pix, eflux_val, eskyflux_val, exptime_val, tp):
    psf = _make_gaussian_psf(sigma_pix)
    nb = NoiseBudget()

    eflux = eflux_val * u.electron / u.second
    eskyflux = eskyflux_val * u.electron / u.second
    exptime = exptime_val * u.second

    snr, _ = nb.get_snr(eflux, psf, eskyflux, exptime)

    recovered_electrons, _ = nb.get_flux_at_snr(
        snr=snr,
        psf=psf,
        eskyflux=eskyflux,
        exptime=exptime,
        signal_throughput=tp,
        coadd_n=1,
    )

    original_electrons = (eflux * exptime).value

    np.testing.assert_allclose(
        recovered_electrons.value * tp,
        original_electrons,
        rtol=1e-6,
        err_msg=(
            f"Round-trip with tp={tp} failed: "
            f"original={original_electrons:.2f}, "
            f"recovered*tp={recovered_electrons.value * tp:.2f}"
        ),
    )


@pytest.mark.parametrize("coadd_n", [1, 2, 5, 10])
def test_coadd_n_round_trip(coadd_n):
    """get_flux_at_snr with coadd_n=N yields per-frame flux that stacks to target SNR.

    Single-frame SNR from get_snr times sqrt(N) should recover the target SNR
    that was passed to get_flux_at_snr.
    """
    psf = _make_gaussian_psf(2.0)
    nb = NoiseBudget()
    eskyflux = 500.0 * u.electron / u.second
    exptime = 30.0 * u.second
    target_snr = 5.0

    above_atm_electrons, _ = nb.get_flux_at_snr(
        snr=target_snr,
        psf=psf,
        eskyflux=eskyflux,
        exptime=exptime,
        signal_throughput=1.0,
        coadd_n=coadd_n,
    )

    single_frame_eflux = above_atm_electrons / exptime

    single_snr, _ = nb.get_snr(
        eflux=single_frame_eflux,
        psf=psf,
        eskyflux=eskyflux,
        exptime=exptime,
    )

    np.testing.assert_allclose(
        single_snr * np.sqrt(coadd_n),
        target_snr,
        rtol=1e-6,
        err_msg=f"coadd_n={coadd_n}: single_snr*sqrt(N)={single_snr * np.sqrt(coadd_n):.6f}, target={target_snr}",
    )


# --- Shared fixtures for end-to-end limiting magnitude test ---

_throughputs_cache = {}


def _get_throughput_and_ab(band):
    """Return cached (SystemThroughput, ABPhot, SpectralSky, band_qe, signal_tp) for a band."""
    if band not in _throughputs_cache:
        collecting_area = config.telescope.collecting_area
        tp = SystemThroughput(throughput_loss=1.0, filters={band: True})
        ab = ABPhot(collecting_area=collecting_area, throughput_loss=1.0, throughput=tp)
        sky = SpectralSky(tp, ab, band)
        band_qe = np.average(tp.qe_freq, weights=tp.filt_freq[band])
        signal_tp = np.average(
            tp.optics[band] * tp.filt_freq.atmosphere,
            weights=tp.filt_freq[band],
        )
        _throughputs_cache[band] = (tp, ab, sky, band_qe, signal_tp)
    return _throughputs_cache[band]


@pytest.mark.parametrize(
    "band,sigma_pix,exptime_val",
    [
        ("g", 1.5, 10),
        ("g", 2.0, 30),
        ("r", 1.0, 60),
        ("r", 2.5, 120),
        ("i", 1.5, 180),
        ("i", 2.0, 300),
    ],
)
def test_exposure_scaling_depth_improvement(band, sigma_pix, exptime_val):
    """Doubling exposure time deepens 5-sigma depth by 1.25*log10(2) mag.

    In the background-limited regime (sky noise >> read noise, dark current),
    SNR ∝ sqrt(t). The faintest detectable source flux therefore scales as
    1/sqrt(t), so depth(2t) - depth(t) = 1.25 * log10(2) ≈ 0.376 mag.
    """
    _, ab, _, band_qe, signal_tp = _get_throughput_and_ab(band)
    psf = _make_gaussian_psf(sigma_pix)
    nb = NoiseBudget(
        read_noise=0.0 * u.electron,
        dark_current=0.0 * u.electron / u.second,
    )

    eskyflux = 1e4 * u.electron / u.second

    def _limiting_mag(t_sec):
        exptime = t_sec * u.second
        above_atm_e, _ = nb.get_flux_at_snr(
            snr=5.0,
            psf=psf,
            eskyflux=eskyflux,
            exptime=exptime,
            signal_throughput=signal_tp,
            coadd_n=1,
        )
        photon_rate = (above_atm_e / exptime) / band_qe
        return ab.mag_from_photons(band, photon_rate)

    depth_t = _limiting_mag(exptime_val)
    depth_2t = _limiting_mag(2 * exptime_val)
    delta = depth_2t - depth_t

    expected = 1.25 * np.log10(2)
    np.testing.assert_allclose(
        delta,
        expected,
        atol=0.02,
        err_msg=(
            f"band={band}, sigma={sigma_pix}, t={exptime_val}s: depth gain={delta:.4f} mag, expected={expected:.4f} mag"
        ),
    )


@given(
    band=sampled_from(["g", "r", "i"]),
    sigma_pix=floats(0.8, 6.0),
    exptime_val=floats(1.0, 300.0),
    eskyflux_val=floats(1e3, 1e5),
)
@settings(max_examples=200)
def test_exposure_scaling_depth_improvement_hypothesis(band, sigma_pix, exptime_val, eskyflux_val):
    """Hypothesis version: doubling exposure deepens depth by 1.25*log10(2) mag.

    Explores the full valid input space rather than 6 hand-chosen points.
    Background-limited regime is enforced by zeroing read noise and dark current.
    """
    _, ab, _, band_qe, signal_tp = _get_throughput_and_ab(band)
    psf = _make_gaussian_psf(sigma_pix)
    nb = NoiseBudget(
        read_noise=0.0 * u.electron,
        dark_current=0.0 * u.electron / u.second,
    )

    eskyflux = eskyflux_val * u.electron / u.second

    def _limiting_mag(t_sec):
        exptime = t_sec * u.second
        above_atm_e, _ = nb.get_flux_at_snr(
            snr=5.0,
            psf=psf,
            eskyflux=eskyflux,
            exptime=exptime,
            signal_throughput=signal_tp,
            coadd_n=1,
        )
        photon_rate = (above_atm_e / exptime) / band_qe
        return ab.mag_from_photons(band, photon_rate)

    depth_t = _limiting_mag(exptime_val)
    depth_2t = _limiting_mag(2 * exptime_val)

    assume(np.isfinite(depth_t) and np.isfinite(depth_2t))

    delta = depth_2t - depth_t
    expected = 1.25 * np.log10(2)

    np.testing.assert_allclose(
        delta,
        expected,
        atol=0.02,
        err_msg=(
            f"band={band}, sigma={sigma_pix}, t={exptime_val}s, sky={eskyflux_val}: "
            f"depth gain={delta:.4f} mag, expected={expected:.4f} mag"
        ),
    )


@pytest.mark.parametrize(
    "band,sigma_pix,area_cm2",
    [
        ("g", 1.5, 1000),
        ("g", 2.0, 5000),
        ("r", 1.0, 2000),
        ("r", 2.5, 10000),
        ("i", 1.5, 8000),
        ("i", 2.0, 20000),
    ],
)
def test_aperture_scaling_depth_improvement(band, sigma_pix, area_cm2):
    """Doubling collecting area deepens 5-sigma depth by 2.5*log10(2) mag.

    With fixed sky electron rate, the noise is unchanged when area doubles.
    The zero point scales linearly with area, so the faintest detectable
    source has half the physical flux density, giving
    depth(2A) - depth(A) = 2.5 * log10(2) ~= 0.752 mag.
    """
    tp, _, _, band_qe, signal_tp = _get_throughput_and_ab(band)
    psf = _make_gaussian_psf(sigma_pix)
    nb = NoiseBudget(
        read_noise=0.0 * u.electron,
        dark_current=0.0 * u.electron / u.second,
    )

    eskyflux = 1e4 * u.electron / u.second
    exptime = 30.0 * u.second

    def _limiting_mag(area):
        ab = ABPhot(collecting_area=area * u.cm**2, throughput_loss=1.0, throughput=tp)
        above_atm_e, _ = nb.get_flux_at_snr(
            snr=5.0,
            psf=psf,
            eskyflux=eskyflux,
            exptime=exptime,
            signal_throughput=signal_tp,
            coadd_n=1,
        )
        photon_rate = (above_atm_e / exptime) / band_qe
        return ab.mag_from_photons(band, photon_rate)

    depth_a = _limiting_mag(area_cm2)
    depth_2a = _limiting_mag(2 * area_cm2)
    delta = depth_2a - depth_a

    expected = 2.5 * np.log10(2)
    np.testing.assert_allclose(
        delta,
        expected,
        atol=0.05,
        err_msg=(
            f"band={band}, sigma={sigma_pix}, A={area_cm2} cm^2: "
            f"depth gain={delta:.4f} mag, expected={expected:.4f} mag"
        ),
    )


@given(
    band=sampled_from(["g", "r", "i"]),
    sigma_pix=floats(0.8, 6.0),
    area_cm2=floats(500.0, 50000.0),
    eskyflux_val=floats(1e3, 1e5),
)
@settings(max_examples=200)
def test_aperture_scaling_depth_improvement_hypothesis(band, sigma_pix, area_cm2, eskyflux_val):
    """Hypothesis version: doubling collecting area deepens depth by 2.5*log10(2) mag.

    Explores the full valid input space rather than 6 hand-chosen points.
    Background-limited regime is enforced by zeroing read noise and dark current.
    """
    tp, _, _, band_qe, signal_tp = _get_throughput_and_ab(band)
    psf = _make_gaussian_psf(sigma_pix)
    nb = NoiseBudget(
        read_noise=0.0 * u.electron,
        dark_current=0.0 * u.electron / u.second,
    )

    eskyflux = eskyflux_val * u.electron / u.second
    exptime = 30.0 * u.second

    def _limiting_mag(area):
        ab = ABPhot(collecting_area=area * u.cm**2, throughput_loss=1.0, throughput=tp)
        above_atm_e, _ = nb.get_flux_at_snr(
            snr=5.0,
            psf=psf,
            eskyflux=eskyflux,
            exptime=exptime,
            signal_throughput=signal_tp,
            coadd_n=1,
        )
        photon_rate = (above_atm_e / exptime) / band_qe
        return ab.mag_from_photons(band, photon_rate)

    depth_a = _limiting_mag(area_cm2)
    depth_2a = _limiting_mag(2 * area_cm2)

    assume(np.isfinite(depth_a) and np.isfinite(depth_2a))

    delta = depth_2a - depth_a
    expected = 2.5 * np.log10(2)

    np.testing.assert_allclose(
        delta,
        expected,
        atol=0.05,
        err_msg=(
            f"band={band}, sigma={sigma_pix}, A={area_cm2} cm^2, sky={eskyflux_val}: "
            f"depth gain={delta:.4f} mag, expected={expected:.4f} mag"
        ),
    )


@pytest.mark.parametrize(
    "band,airglow_s10,zodiacal_s10,starlight_s10,moon_s10,artif_s10,sigma_pix,exptime_val,snr_threshold",
    [
        ("g", 50, 30, 20, 0, 0, 1.5, 30, 5.0),
        ("g", 75, 45, 30, 50, 5, 2.0, 10, 5.0),
        ("r", 50, 30, 20, 0, 0, 1.0, 30, 5.0),
        ("r", 100, 60, 40, 100, 10, 3.0, 60, 5.0),
        ("i", 50, 30, 20, 0, 0, 1.5, 30, 5.0),
        ("i", 60, 36, 24, 30, 2, 2.5, 15, 5.0),
        ("g", 40, 24, 16, 20, 1, 1.0, 1, 5.0),
    ],
)
def test_limiting_magnitude_self_consistency(
    band, airglow_s10, zodiacal_s10, starlight_s10, moon_s10, artif_s10, sigma_pix, exptime_val, snr_threshold
):
    """A source at the computed limiting magnitude achieves exactly the detection SNR.

    For any valid (band, sky, PSF, exptime), compute the limiting magnitude via
    the survey pipeline path (get_flux_at_snr -> mag_from_photons), then verify
    that converting back (photons_from_mag -> get_snr) recovers the detection
    threshold SNR. This spans ABPhot, NoiseBudget, and SpectralSky.
    """
    _, ab, sky, band_qe, signal_tp = _get_throughput_and_ab(band)

    psf = _make_gaussian_psf(sigma_pix)
    nb = NoiseBudget(read_noise=config.telescope.readnoise * u.electron)
    exptime = exptime_val * u.second

    # Sky electron rate per pixel from spectral sky model
    sky_e_per_arcsec2 = sky.sky_electrons(airglow_s10, zodiacal_s10, starlight_s10, moon_s10, artif_s10)
    plate_scale = config.telescope.plate_scale
    eskyflux = sky_e_per_arcsec2 * plate_scale**2 * u.electron / u.second

    # Forward path: compute limiting magnitude
    above_atm_electrons, _ = nb.get_flux_at_snr(
        snr=snr_threshold,
        psf=psf,
        eskyflux=eskyflux,
        exptime=exptime,
        signal_throughput=signal_tp,
        coadd_n=1,
    )
    signal_e_rate = above_atm_electrons / exptime
    signal_photon_rate = signal_e_rate / band_qe
    limmag = ab.mag_from_photons(band, signal_photon_rate)

    # Reverse path: convert limiting magnitude back to flux and compute SNR
    recovered_photon_rate = ab.photons_from_mag(band, limmag)
    recovered_electron_rate = recovered_photon_rate * band_qe
    detected_electron_rate = recovered_electron_rate * signal_tp

    recovered_snr, _ = nb.get_snr(
        eflux=detected_electron_rate,
        psf=psf,
        eskyflux=eskyflux,
        exptime=exptime,
    )

    np.testing.assert_allclose(
        recovered_snr,
        snr_threshold,
        rtol=1e-6,
        err_msg=(
            f"band={band}, sky=({airglow_s10},{zodiacal_s10},{starlight_s10},{moon_s10},{artif_s10}), "
            f"sigma={sigma_pix}, t={exptime_val}s: "
            f"SNR={recovered_snr:.6f}, expected={snr_threshold}"
        ),
    )
