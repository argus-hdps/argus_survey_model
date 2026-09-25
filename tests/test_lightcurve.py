"""Properties of argus_sim.lightcurve.

Unit properties run on synthetic visit tables; the run-level properties need a real exact-coverage run, named by
the environment variable ARGUS_LC_RUN (for example the output of ``asim survey --start 2026-06-14T12:00
--n-nights 1 --seed 1 --no-weather --outdir RUN``). They skip without it and when the run does not match this
package (see conftest.py).
"""

import os

import astropy.table as tbl
import numpy as np
import pytest

from argus_sim.lightcurve import SurveyRun, inject, limit_mag, sigma_mag, snr

ZP = {
    "g": {"zp_photons_per_sec": 799614287.18, "band_qe": 0.79196},
    "rho": {"zp_photons_per_sec": 836648274.55, "band_qe": 0.45953},
}
MAG_ERR_PER_INV_SNR = 2.5 / np.log(10)


def _visits(n=40, fast=False, seed=0):
    rng = np.random.default_rng(seed)
    return tbl.Table(
        {
            "mjd": 62000.0 + np.arange(n) / 1440.0,
            "band": np.where(np.arange(n) % 2 == 0, "g", "rho"),
            "tel_id": rng.integers(0, 1200, n),
            "subarray": rng.integers(0, 8, n),
            "exptime_s": np.full(n, 1.0 if fast else 60.0),
            "n_frames": np.full(n, 60 if fast else 1, dtype=int),
            "sigma_bkg_e": rng.uniform(12.0, 14.0, n) if fast else rng.uniform(35.0, 50.0, n),
            "throughput": rng.uniform(0.45, 0.62, n),
            "airmass": rng.uniform(1.0, 1.6, n),
            "field_angle": rng.uniform(0.0, 2.0, n),
            "seeing_zenith": np.full(n, 0.9),
            "transparency": np.ones(n),
            "ratchet": np.zeros(n, dtype=int),
            "night": np.full(n, 61999, dtype=int),
        }
    )


def _flux_e(vis, mag):
    zp = np.array([ZP[b]["zp_photons_per_sec"] for b in vis["band"]])
    qe = np.array([ZP[b]["band_qe"] for b in vis["band"]])
    return zp * 10 ** (-0.4 * np.asarray(mag)) * qe * np.asarray(vis["exptime_s"]) * np.asarray(vis["throughput"])


# --- the photometric model -------------------------------------------------------------------------------------


def test_snr_matches_the_specified_model():
    vis = _visits()
    s = _flux_e(vis, 17.0)
    expected = s / np.sqrt(np.asarray(vis["sigma_bkg_e"]) ** 2 + s)
    np.testing.assert_allclose(snr(vis, 17.0, ZP), expected, rtol=1e-10)
    np.testing.assert_allclose(snr(vis, 17.0, ZP, source_poisson=False), s / vis["sigma_bkg_e"], rtol=1e-10)


def test_sigma_mag_grows_monotonically_with_magnitude():
    vis = _visits()
    mags = np.linspace(8.0, 24.0, 60)
    sig = np.array([sigma_mag(vis, m, ZP) for m in mags])
    assert np.all(np.diff(sig, axis=0) > 0)


def test_bright_limit_is_the_systematic_floor():
    vis = _visits()
    np.testing.assert_allclose(sigma_mag(vis, 4.0, ZP), 0.005, rtol=2e-3)
    np.testing.assert_allclose(sigma_mag(vis, 4.0, ZP, floor=0.0), 0.0, atol=1e-3)


def test_floor_adds_in_quadrature():
    vis = _visits()
    for m in (12.0, 18.0, 21.0):
        a = sigma_mag(vis, m, ZP, floor=0.0)
        b = sigma_mag(vis, m, ZP, floor=0.005)
        np.testing.assert_allclose(b**2, a**2 + 0.005**2, rtol=1e-10)


def test_faint_limit_is_background_dominated():
    vis = _visits()
    ratio = snr(vis, 25.0, ZP) / snr(vis, 25.0, ZP, source_poisson=False)
    np.testing.assert_allclose(ratio, 1.0, atol=1e-2)
    np.testing.assert_allclose(
        sigma_mag(vis, 25.0, ZP, floor=0.0), MAG_ERR_PER_INV_SNR / snr(vis, 25.0, ZP), rtol=1e-10
    )


def test_limit_mag_is_where_snr_equals_nsigma():
    vis = _visits()
    for nsig in (3.0, 5.0):
        lim = limit_mag(vis, ZP, nsigma=nsig)
        np.testing.assert_allclose(snr(vis, lim, ZP), nsig, rtol=1e-6)


def test_limit_mag_reproduces_the_survey_depth_convention():
    """Background-only limit of one exposure == tanseg_coverage.minipix_limmag with ivar = T^2/sigma^2, nobs 1."""
    from argus_sim.tanseg_coverage import minipix_limmag

    class _AB:
        def mag_from_photons(self, band, photons):
            return -2.5 * np.log10(np.asarray(getattr(photons, "value", photons)) / ZP[band]["zp_photons_per_sec"])

    vis = _visits()
    lim = limit_mag(vis, ZP, nsigma=5.0, source_poisson=False)
    for i, row in enumerate(vis):
        ivar = np.array([row["throughput"] ** 2 / row["sigma_bkg_e"] ** 2])
        ref = minipix_limmag(ivar, np.ones(1), None, _AB(), row["band"], ZP[row["band"]]["band_qe"], 5.0, 60.0, 1.0)
        np.testing.assert_allclose(lim[i], ref[0], rtol=1e-9)


def test_one_second_frames_are_sqrt60_shallower_than_a_minute_when_background_limited():
    base, fast = _visits(), _visits(fast=True)
    fast["sigma_bkg_e"] = base["sigma_bkg_e"] / np.sqrt(60.0)  # same sky, 1/60 of the background electrons
    ratio = snr(base, 24.0, ZP, source_poisson=False, per="frame") / snr(
        fast, 24.0, ZP, source_poisson=False, per="frame"
    )
    np.testing.assert_allclose(ratio, np.sqrt(60.0), rtol=1e-10)


# --- injection -------------------------------------------------------------------------------------------------


def test_inject_is_deterministic_for_a_seed():
    vis = _visits()
    a = inject(vis, lambda t, b: np.full(len(t), 18.0), ZP, rng=np.random.default_rng(7))
    b = inject(vis, lambda t, b: np.full(len(t), 18.0), ZP, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a["flux_obs_e"], b["flux_obs_e"])


def test_inject_constant_source_pulls_are_unit_normal():
    vis = _visits(n=400, seed=3)
    out = inject(vis, lambda t, b: np.full(len(t), 17.5), ZP, rng=np.random.default_rng(11), cadence="slot")
    pull = (out["flux_obs_e"] - out["flux_true_e"]) / out["flux_err_e"]
    assert abs(np.mean(pull)) < 0.2
    assert 0.85 < np.std(pull) < 1.15
    np.testing.assert_allclose(out["mag_true"], 17.5)


def test_inject_follows_the_model_and_flags_detections():
    vis = _visits(n=200, seed=4)
    model = lambda t, b: 15.0 + 8.0 * ((t - t.min()) / (t.max() - t.min()))  # noqa: E731, 15 -> 23 mag ramp
    out = inject(vis, model, ZP, rng=np.random.default_rng(5), cadence="slot")
    np.testing.assert_allclose(out["mag_true"], model(np.asarray(vis["mjd"]), np.asarray(vis["band"])))
    np.testing.assert_array_equal(out["detected"], out["flux_obs_e"] / out["flux_err_e"] >= 5.0)
    assert out["detected"][:20].all() and not out["detected"][-20:].any()
    assert np.all(np.isnan(out["mag_obs"][out["flux_obs_e"] <= 0]))
    np.testing.assert_allclose(out["limit_mag"], limit_mag(vis, ZP, nsigma=5.0), rtol=1e-9)


def test_native_cadence_expands_fast_slots_to_sixty_one_second_frames():
    vis = tbl.vstack([_visits(n=4), _visits(n=3, fast=True)])
    out = inject(vis, lambda t, b: np.full(len(t), 16.0), ZP, rng=np.random.default_rng(1), cadence="native")
    assert len(out) == 4 + 3 * 60
    fast = out[out["exptime_s"] == 1.0]
    dt = np.diff(np.sort(np.asarray(fast["mjd"])[:60])) * 86400.0
    np.testing.assert_allclose(dt, 1.0, atol=1e-3)
    slot = inject(vis, lambda t, b: np.full(len(t), 16.0), ZP, rng=np.random.default_rng(1), cadence="slot")
    assert len(slot) == len(vis)


# --- a real run ------------------------------------------------------------------------------------------------

RUN = os.path.expanduser(os.environ.get("ARGUS_LC_RUN", ""))
needs_run = pytest.mark.needs_lc_run


@pytest.fixture(scope="module")
def run(lc_index):
    return SurveyRun(RUN, index_dir=lc_index)


def _covered_points(n, seed=0):
    """Owned minipix centres that the run covered at least once (from the survey's own totals)."""
    from argus_sim import skymap_shim as sh

    nobs = np.load(os.path.join(RUN, "minipix", "total_nobs.npy"), mmap_mode="r")
    tid = np.load(os.path.join(RUN, "minipix", "tanseg_id.npy"))
    rng = np.random.default_rng(seed)
    rows = rng.choice(np.flatnonzero(np.asarray(nobs[0]).sum(1) > 0), size=4 * n, replace=False)
    xi, eta = sh.minipix_offsets()
    out = []
    for r in rows:
        mp = int(rng.integers(0, 4096))
        T, E, N = sh.tangent_basis(*sh.tile_center(tid[r : r + 1]))
        p = T[0] + xi[mp] * E[0] + eta[mp] * N[0]
        p = p / np.linalg.norm(p)
        ra, dec = np.degrees(np.arctan2(p[1], p[0])) % 360.0, np.degrees(np.arcsin(p[2]))
        t2, m2 = sh.radec_to_composite_id(np.array([ra]), np.array([dec]))
        if int(t2[0]) == int(tid[r]) and int(m2[0]) == mp:
            out.append((ra, dec, r, mp))
        if len(out) == n:
            break
    return out


@needs_run
def test_visits_reproduce_the_surveys_own_totals(run):
    """Sum of n_frames * T^2/sigma^2 over a point's visits == total_ivar; frame counts == total_nobs / fast units."""
    ivar = np.load(os.path.join(RUN, "minipix", "total_ivar.npy"), mmap_mode="r")
    nobs = np.load(os.path.join(RUN, "minipix", "total_nobs.npy"), mmap_mode="r")
    fpath = os.path.join(RUN, "minipix", "total_fast_units.npy")
    fast = np.load(fpath, mmap_mode="r") if os.path.isfile(fpath) else None
    for ra, dec, r, mp in _covered_points(25):
        vis = run.visits(ra, dec)
        for b, band in enumerate(run.bands):
            v = vis[vis["band"] == band]
            w = np.asarray(v["n_frames"]) * np.asarray(v["throughput"]) ** 2 / np.asarray(v["sigma_bkg_e"]) ** 2
            np.testing.assert_allclose(w.sum(), ivar[b, r, mp], rtol=1e-4, atol=1e-12)
            base = int(np.sum(np.asarray(v["n_frames"])[np.asarray(v["exptime_s"]) == 60.0]))
            assert base == int(nobs[b, r, mp])
            n1 = int(np.sum(np.asarray(v["n_frames"])[np.asarray(v["exptime_s"]) == 1.0]))
            assert n1 == (int(fast[b, r, mp]) * 60 if fast is not None else 0)


@needs_run
def test_visit_table_shape_and_order(run):
    ra, dec, _, _ = _covered_points(1, seed=1)[0]
    vis = run.visits(ra, dec)
    assert len(vis) > 0
    for col in (
        "mjd",
        "band",
        "tel_id",
        "subarray",
        "exptime_s",
        "n_frames",
        "sigma_bkg_e",
        "throughput",
        "airmass",
        "field_angle",
        "seeing_zenith",
        "transparency",
        "ratchet",
        "night",
    ):
        assert col in vis.colnames
    order = np.lexsort((np.asarray(vis["tel_id"]), np.asarray(vis["mjd"])))
    np.testing.assert_array_equal(order, np.arange(len(vis)))
    assert set(vis["band"]) <= set(run.bands)
    assert np.all(np.isin(vis["n_frames"], [1, 60]))


@needs_run
def test_point_outside_the_footprint_is_empty_with_columns(run):
    vis = run.visits(10.0, -70.0)
    assert len(vis) == 0 and "sigma_bkg_e" in vis.colnames


@needs_run
def test_visits_many_matches_single_queries(run):
    pts = _covered_points(3, seed=2)
    many = run.visits_many(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    for i, (ra, dec, _, _) in enumerate(pts):
        one = run.visits(ra, dec)
        sub = many[many["source"] == i]
        np.testing.assert_array_equal(np.asarray(sub["mjd"]), np.asarray(one["mjd"]))
