"""Property tests for filter depth calculations."""

import itertools

import astropy.time as atime
import astropy.units as u
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from argus_sim import ABPhot, GaussianPSF, NoiseBudget, SpectralSky, SystemThroughput, c
from argus_sim.observatory import Observatory


def flux_average_mags(mags):
    """Average magnitudes in flux space.

    Converts to flux (10^(-0.4*m)), averages, converts back.
    """
    fluxes = 10 ** (-0.4 * mags)
    return -2.5 * np.log10(np.mean(fluxes))


def flux_average_mags_nan(mags):
    """Flux-space average that ignores NaNs."""
    fluxes = 10 ** (-0.4 * mags)
    return -2.5 * np.log10(np.nanmean(fluxes))


mag_arrays = st.lists(
    st.floats(min_value=10.0, max_value=30.0, allow_nan=False, allow_infinity=False),
    min_size=2,
    max_size=100,
).filter(lambda mags: np.std(mags) > 0.01)


@given(mags=mag_arrays)
@settings(max_examples=200)
def test_flux_average_brighter_than_mag_average(mags):
    """Flux-averaged depth is brighter (smaller mag) than mag-averaged depth.

    By Jensen's inequality on the convex function 10^(-0.4*m), the mean
    flux exceeds the flux at the mean magnitude, so the flux-averaged
    magnitude is smaller (brighter / more conservative).
    """
    arr = np.array(mags)
    mag_avg = np.mean(arr)
    flux_avg = flux_average_mags(arr)

    assert flux_avg < mag_avg, (
        f"Expected flux_avg ({flux_avg:.6f}) < mag_avg ({mag_avg:.6f}) for mags with std={np.std(arr):.4f}"
    )


@given(mags=mag_arrays)
@settings(max_examples=200)
def test_flux_average_equals_mag_average_when_uniform(mags):
    """When all magnitudes are identical, both averages agree."""
    val = mags[0]
    arr = np.full(len(mags), val)
    mag_avg = np.mean(arr)
    flux_avg = flux_average_mags(arr)
    assert np.isclose(flux_avg, mag_avg, atol=1e-10)


@given(mags=mag_arrays)
@settings(max_examples=200)
def test_flux_average_nan_matches_clean(mags):
    """NaN-aware flux average on clean data matches the standard version."""
    arr = np.array(mags)
    assert np.isclose(flux_average_mags(arr), flux_average_mags_nan(arr), atol=1e-10)


def _synth_depth_bkg_only(sig_broad_rate, sig_narrow_rate, bkg_broad, bkg_narrow, sharpness, snr, ref_mag):
    """Synthetic depth with background-only noise (no source shot noise)."""
    noise_broad = np.sqrt(bkg_broad / sharpness)
    noise_narrow = np.sqrt(bkg_narrow / sharpness)
    combined_noise = np.sqrt(noise_broad**2 + noise_narrow**2)
    synth_signal_ref = sig_broad_rate - sig_narrow_rate
    if synth_signal_ref <= 0:
        return np.nan
    required_signal = snr * combined_noise
    return ref_mag - 2.5 * np.log10(required_signal / synth_signal_ref)


def _synth_depth_with_source(sig_broad_rate, sig_narrow_rate, bkg_broad, bkg_narrow, sharpness, snr, ref_mag):
    """Synthetic depth including source shot noise via the quadratic formula.

    Solves SNR = D*f / sqrt(P*f + B) for f, where:
      D = sig_broad_rate - sig_narrow_rate  (differential signal at ref_mag)
      P = sig_broad_rate + sig_narrow_rate  (total source electrons at ref_mag)
      B = (bkg_broad + bkg_narrow) / sharpness
      f = 10^(-0.4*(lim_mag - ref_mag))
    """
    D = sig_broad_rate - sig_narrow_rate
    if D <= 0:
        return np.nan
    P = sig_broad_rate + sig_narrow_rate
    B = (bkg_broad + bkg_narrow) / sharpness

    discriminant = snr**4 * P**2 + 4 * D**2 * snr**2 * B
    f = (snr**2 * P + np.sqrt(discriminant)) / (2 * D**2)

    return ref_mag - 2.5 * np.log10(f)


positive_float = st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False)


@given(
    sig_broad_rate=positive_float,
    sig_narrow_frac=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
    bkg_broad=positive_float,
    bkg_narrow=positive_float,
    sharpness=st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False),
    snr=st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=500)
def test_source_shot_noise_makes_synthetic_depth_shallower(
    sig_broad_rate, sig_narrow_frac, bkg_broad, bkg_narrow, sharpness, snr
):
    """Synthetic depth with source shot noise is shallower (brighter limiting mag).

    Source shot noise adds variance, so more flux is needed for the same
    SNR. More flux means a brighter source, so the limiting magnitude
    (faintest detectable) is a smaller number.
    """
    ref_mag = 20.0
    sig_narrow_rate = sig_broad_rate * sig_narrow_frac

    depth_bkg = _synth_depth_bkg_only(sig_broad_rate, sig_narrow_rate, bkg_broad, bkg_narrow, sharpness, snr, ref_mag)
    depth_src = _synth_depth_with_source(
        sig_broad_rate, sig_narrow_rate, bkg_broad, bkg_narrow, sharpness, snr, ref_mag
    )

    if np.isnan(depth_bkg) or np.isnan(depth_src):
        return

    assert depth_src <= depth_bkg + 1e-10, (
        f"Source shot noise should make depth shallower (smaller mag number): "
        f"with_source={depth_src:.4f} > bkg_only={depth_bkg:.4f}"
    )


@given(
    sig_broad_rate=positive_float,
    sig_narrow_frac=st.floats(min_value=0.01, max_value=0.7, allow_nan=False, allow_infinity=False),
    bkg_broad=st.floats(min_value=1e6, max_value=1e8, allow_nan=False, allow_infinity=False),
    bkg_narrow=st.floats(min_value=1e6, max_value=1e8, allow_nan=False, allow_infinity=False),
    sharpness=st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False),
    snr=st.floats(min_value=3.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_source_shot_noise_converges_to_bkg_only_in_sky_dominated_regime(
    sig_broad_rate, sig_narrow_frac, bkg_broad, bkg_narrow, sharpness, snr
):
    """When background dominates, source shot noise correction is small.

    In the sky-dominated regime (large bkg relative to signal), the source
    contribution to noise is negligible and both depths should nearly agree.
    Restricts narrow_frac < 0.7 so the differential signal is not pathologically
    small relative to the sum (which would amplify the source noise term).
    """
    ref_mag = 20.0
    sig_narrow_rate = sig_broad_rate * sig_narrow_frac

    depth_bkg = _synth_depth_bkg_only(sig_broad_rate, sig_narrow_rate, bkg_broad, bkg_narrow, sharpness, snr, ref_mag)
    depth_src = _synth_depth_with_source(
        sig_broad_rate, sig_narrow_rate, bkg_broad, bkg_narrow, sharpness, snr, ref_mag
    )

    if np.isnan(depth_bkg) or np.isnan(depth_src):
        return

    diff = depth_bkg - depth_src
    assert diff >= -1e-10, f"Source depth should be <= bkg depth, got diff={diff:.6f}"
    assert diff < 0.5, f"In sky-dominated regime, source shot noise correction should be < 0.5 mag, got {diff:.4f} mag"


RANK_FILTERS = ["A", "g", "r", "i", "r+i+z", "i+z"]


@pytest.fixture(scope="module")
def system_response_matrix():
    """Build the 6 x N_wavelength system response matrix.

    Each row is filter_transmission * optics * QE * atmosphere evaluated
    on the standard wavelength grid.
    """
    tp = SystemThroughput(
        filters={f: True for f in RANK_FILTERS},
    )
    rows = []
    for filt in RANK_FILTERS:
        response = (
            np.asarray(tp.filt_wav[filt]) * tp.optics[filt] * np.asarray(tp.qe_wav) * np.asarray(tp.filt_wav.atmosphere)
        )
        rows.append(response)
    return np.vstack(rows)


def _all_nonempty_subsets(n):
    """Yield all 2^n - 1 non-empty subsets as tuples of indices."""
    for k in range(1, n + 1):
        yield from itertools.combinations(range(n), k)


def _matrix_rank_svd(matrix, tol=1e-10):
    """Rank via SVD with absolute tolerance on singular values."""
    sv = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(sv > tol))


@pytest.mark.parametrize(
    "subset",
    list(_all_nonempty_subsets(len(RANK_FILTERS))),
    ids=lambda s: "+".join(RANK_FILTERS[i] for i in s),
)
def test_matrix_rank_equals_subset_size(system_response_matrix, subset):
    """Any K-filter subset produces a system response matrix of rank K.

    Linear independence of the system response vectors guarantees that
    the filter set provides K independent spectral measurements, which
    is the minimum requirement for K-band photometric classification.
    """
    sub_matrix = system_response_matrix[list(subset), :]
    k = len(subset)
    rank = _matrix_rank_svd(sub_matrix)
    assert rank == k, f"Subset {{{', '.join(RANK_FILTERS[i] for i in subset)}}} has rank {rank}, expected {k}"


# ---------------------------------------------------------------------------
# Depth ranking property test: A > g > i across observing conditions
# ---------------------------------------------------------------------------

DEPTH_BANDS = ["A", "g", "r", "i"]
_THROUGHPUT_LOSS_REF = 0.75
_EXPTIME = 60.0
_SNR = 5.0


@pytest.fixture(scope="module")
def depth_model():
    """Pre-compute the instrument model and sky grid for depth ranking tests.

    Builds SystemThroughput, ABPhot, SpectralSky, NoiseBudget, and the
    Observatory at a reference throughput_loss.  Pre-computes sky S_10
    components on a grid of altitudes at a dark (new-moon) epoch so that
    individual Hypothesis trials only need cheap interpolation.
    """
    tp = SystemThroughput(
        throughput_loss=_THROUGHPUT_LOSS_REF,
        filters={b: True for b in DEPTH_BANDS},
    )

    collecting_area = c.telescope.collecting_area

    ab = ABPhot(
        collecting_area=collecting_area,
        throughput_loss=_THROUGHPUT_LOSS_REF,
        throughput=tp,
    )

    psf_gen = GaussianPSF(
        plate_scale=c.telescope.plate_scale,
        pixel_size=c.telescope.pixel_size,
        rms_spot_size=c.telescope.spot_size,
    )

    nb = NoiseBudget(
        read_noise=c.telescope.readnoise * u.electron,
        dark_current=0.0022 * u.electron / u.second,
    )

    plate_scale = c.telescope.plate_scale

    signal_tp = {}
    band_qe = {}
    zp_photons = {}
    for band in DEPTH_BANDS:
        signal_tp[band] = np.average(
            tp.optics[band] * tp.filt_freq.atmosphere,
            weights=tp.filt_freq[band],
        )
        band_qe[band] = np.average(tp.qe_freq, weights=tp.filt_freq[band]).value
        zp_photons[band] = ab.photons_from_flux_density(band, ab.zp_ergs)

    sky_models = {band: SpectralSky(tp, ab, band) for band in DEPTH_BANDS}

    obs = Observatory()
    dark_epoch = atime.Time("2025-03-29T06:00:00", scale="utc")
    bright_epoch = atime.Time("2025-01-15T03:00:00", scale="utc")

    alt_grid = np.linspace(33.0, 90.0, 30)
    az_fixed = 180.0
    sky_grid = {}
    bright_sky_grid = {}
    for alt_val in alt_grid:
        sky_grid[alt_val] = obs.sky_components_at(alt_val, az_fixed, dark_epoch)
        bright_sky_grid[alt_val] = obs.sky_components_at(alt_val, az_fixed, bright_epoch)

    return {
        "signal_tp": signal_tp,
        "band_qe": band_qe,
        "zp_photons": zp_photons,
        "sky_models": sky_models,
        "psf_gen": psf_gen,
        "nb": nb,
        "plate_scale": plate_scale,
        "alt_grid": alt_grid,
        "sky_grid": sky_grid,
        "bright_sky_grid": bright_sky_grid,
    }


def _limmag_for_band(band, model, seeing, throughput_loss, sky_comp):
    """Compute 60s 5-sigma limiting magnitude for a single band and condition.

    Accepts a throughput_loss that may differ from the reference model's
    built-in loss.  Scales signal_tp and sky rates by the ratio instead of
    rebuilding the full instrument model per trial.
    """
    loss_ratio = throughput_loss / _THROUGHPUT_LOSS_REF

    psf, _ = model["psf_gen"].with_seeing(seeing)
    sky_e_per_arcsec2 = model["sky_models"][band].sky_electrons(
        sky_comp["airglow_s10"],
        sky_comp["zodiacal_s10"],
        sky_comp["starlight_s10"],
        sky_comp["moon_s10"],
        sky_comp["artif_s10"],
    )
    sky_e_per_arcsec2 = sky_e_per_arcsec2 * loss_ratio
    sky_e_per_pix = sky_e_per_arcsec2 * model["plate_scale"] ** 2

    scaled_signal_tp = model["signal_tp"][band] * loss_ratio

    signal_e, _ = model["nb"].get_flux_at_snr(
        snr=_SNR,
        psf=psf,
        eskyflux=sky_e_per_pix * u.electron / u.second,
        exptime=_EXPTIME * u.second,
        signal_throughput=scaled_signal_tp,
    )

    photon_rate = (signal_e.value / model["band_qe"][band] / _EXPTIME) * u.photon / u.second
    mag = -2.5 * np.log10(photon_rate / model["zp_photons"][band])
    return float(mag)


def _nearest_sky(model, alt):
    """Look up pre-computed dark-sky components for the nearest grid altitude."""
    grid = model["alt_grid"]
    idx = np.argmin(np.abs(grid - alt))
    return model["sky_grid"][grid[idx]]


def _nearest_bright_sky(model, alt):
    """Look up pre-computed bright-sky (full-moon) components for the nearest grid altitude."""
    grid = model["alt_grid"]
    idx = np.argmin(np.abs(grid - alt))
    return model["bright_sky_grid"][grid[idx]]


@given(
    seeing=st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False),
    throughput_loss=st.floats(min_value=0.6, max_value=0.9, allow_nan=False, allow_infinity=False),
    alt=st.floats(min_value=33.0, max_value=90.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200, deadline=None)
def test_depth_ranking_A_gt_g_gt_i(depth_model, seeing, throughput_loss, alt):
    """Depth ordering A > g > i holds across observing conditions.

    The A (open) filter has the widest spectral acceptance, collecting more
    photons from a flat-spectrum source than g, which in turn is wider than i.
    More collected photons means fainter detectable sources (larger limiting
    magnitude).  The ordering holds over the tested ranges of seeing,
    throughput loss and sky brightness.
    """
    sky_comp = _nearest_sky(depth_model, alt)

    depth_A = _limmag_for_band("A", depth_model, seeing, throughput_loss, sky_comp)
    depth_g = _limmag_for_band("g", depth_model, seeing, throughput_loss, sky_comp)
    depth_i = _limmag_for_band("i", depth_model, seeing, throughput_loss, sky_comp)

    assert depth_A > depth_g, (
        f"Expected depth(A)={depth_A:.3f} > depth(g)={depth_g:.3f} "
        f'[seeing={seeing:.2f}", loss={throughput_loss:.2f}, alt={alt:.1f}°]'
    )
    assert depth_g > depth_i, (
        f"Expected depth(g)={depth_g:.3f} > depth(i)={depth_i:.3f} "
        f'[seeing={seeing:.2f}", loss={throughput_loss:.2f}, alt={alt:.1f}°]'
    )


@given(
    seeing=st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False),
    throughput_loss=st.floats(min_value=0.6, max_value=0.9, allow_nan=False, allow_infinity=False),
    alt=st.floats(min_value=33.0, max_value=90.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200, deadline=None)
def test_dark_sky_gain_monotonic_gri(depth_model, seeing, throughput_loss, alt):
    """Dark-sky depth gain follows gain(g) > gain(r) > gain(i).

    Bluer bands suffer more sky background increase from moonlight because
    lunar scattered light peaks in the blue.  The depth *gain* from going
    dark (new moon minus full moon) is therefore largest for g and
    smallest for i.  The A filter spans blue+red so it does not follow
    simple wavelength ordering; only g/r/i are tested.
    """
    dark_comp = _nearest_sky(depth_model, alt)
    bright_comp = _nearest_bright_sky(depth_model, alt)

    gains = {}
    for band in ("g", "r", "i"):
        depth_dark = _limmag_for_band(band, depth_model, seeing, throughput_loss, dark_comp)
        depth_bright = _limmag_for_band(band, depth_model, seeing, throughput_loss, bright_comp)
        gains[band] = depth_dark - depth_bright

    assert gains["g"] > gains["r"], (
        f"Expected gain(g)={gains['g']:.4f} > gain(r)={gains['r']:.4f} "
        f'[seeing={seeing:.2f}", loss={throughput_loss:.2f}, alt={alt:.1f}°]'
    )
    assert gains["r"] > gains["i"], (
        f"Expected gain(r)={gains['r']:.4f} > gain(i)={gains['i']:.4f} "
        f'[seeing={seeing:.2f}", loss={throughput_loss:.2f}, alt={alt:.1f}°]'
    )
