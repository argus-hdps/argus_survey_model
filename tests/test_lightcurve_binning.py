"""argus_sim.lightcurve: binning, depth maps, night logs and model injection.

Unit properties use synthetic visit tables; run-level properties need a real run named by ARGUS_LC_RUN (see
test_lightcurve.py) and skip without it.
"""

import inspect
import os

import astropy.table as tbl
import numpy as np
import pytest

from argus_sim.lightcurve import SurveyRun, bin_lightcurve, inject, limit_mag, sigma_mag, snr

ZP = {
    "g": {"zp_photons_per_sec": 799614287.18, "band_qe": 0.79196},
    "rho": {"zp_photons_per_sec": 836648274.55, "band_qe": 0.45953},
}


def _visits(n=40, fast=False, seed=0, nights=1, slots_per_ratchet=15):
    rng = np.random.default_rng(seed)
    k = np.arange(n)
    return tbl.Table(
        {
            "mjd": 62000.0 + (k // max(n // nights, 1)) + (k % max(n // nights, 1)) / 1440.0,
            "band": np.where(k % 2 == 0, "g", "rho"),
            "tel_id": rng.integers(0, 1200, n),
            "subarray": rng.integers(0, 8, n),
            "exptime_s": np.full(n, 1.0 if fast else 60.0),
            "n_frames": np.full(n, 60 if fast else 1, dtype=int),
            "sigma_bkg_e": rng.uniform(12.0, 14.0, n) if fast else rng.uniform(35.0, 50.0, n),
            "throughput": rng.uniform(0.45, 0.62, n),
            "airmass": rng.uniform(1.0, 1.6, n),
            "field_angle": rng.uniform(0.0, 2.0, n),
            "seeing_zenith": rng.uniform(0.7, 1.4, n),
            "transparency": np.ones(n),
            "ratchet": (k // slots_per_ratchet).astype(int),
            "night": (61999 + k // max(n // nights, 1)).astype(int),
        }
    )


def _const(mag):
    return lambda t, b: np.full(len(t), float(mag))


# --- defaults and the frame / slot switch -----------------------------------------------------------------------


def test_inject_defaults_to_one_row_per_slot():
    assert inspect.signature(inject).parameters["cadence"].default == "slot"
    vis = tbl.vstack([_visits(n=6), _visits(n=4, fast=True)])
    out = inject(vis, _const(16.0), ZP, rng=np.random.default_rng(0))
    assert len(out) == len(vis)


def test_per_slot_is_the_default_and_matches_inject():
    """limit_mag(vis) and inject(...)['limit_mag'] describe the same measurement: one slot."""
    vis = tbl.vstack([_visits(n=6), _visits(n=6, fast=True)])
    for f in (snr, sigma_mag, limit_mag):
        assert inspect.signature(f).parameters["per"].default == "slot"
        assert "stack" not in inspect.signature(f).parameters
    out = inject(vis, _const(16.0), ZP, rng=np.random.default_rng(0))
    np.testing.assert_allclose(out["limit_mag"], limit_mag(vis, ZP), rtol=1e-9)


def test_per_frame_and_per_slot_differ_only_for_fast_slots():
    base, fast = _visits(n=10), _visits(n=10, fast=True)
    np.testing.assert_allclose(limit_mag(base, ZP, per="frame"), limit_mag(base, ZP, per="slot"), rtol=1e-12)
    gain = limit_mag(fast, ZP, per="slot", source_poisson=False) - limit_mag(
        fast, ZP, per="frame", source_poisson=False
    )
    np.testing.assert_allclose(gain, 2.5 * np.log10(np.sqrt(60.0)), rtol=1e-9)  # background-limited: sqrt(60)
    with pytest.raises(ValueError):
        limit_mag(base, ZP, per="minute")


# --- fluxes in microjansky ---------------------------------------------------------------------------------------


def test_inject_reports_flux_in_microjansky():
    vis = tbl.vstack([_visits(n=8), _visits(n=8, fast=True)])
    out = inject(vis, _const(18.9), ZP, rng=np.random.default_rng(2))
    np.testing.assert_allclose(out["flux_true_ujy"], 10 ** ((23.9 - 18.9) / 2.5), rtol=1e-9)  # AB 23.9 = 1 uJy
    ratio = np.asarray(out["flux_obs_ujy"]) / np.asarray(out["flux_obs_e"])
    np.testing.assert_allclose(np.asarray(out["flux_err_ujy"]) / np.asarray(out["flux_err_e"]), ratio, rtol=1e-9)


# --- binning -----------------------------------------------------------------------------------------------------


def test_binning_is_unbiased_for_a_source_below_the_single_visit_limit():
    """Averaging in flux over all measurements recovers a source too faint to detect in one visit."""
    vis = _visits(n=6000, seed=5, nights=100)
    true_mag = 21.0  # about 2 sigma per 60-s g visit, 1 sigma in rho
    lc = inject(vis, _const(true_mag), ZP, rng=np.random.default_rng(6))
    assert lc["detected"].mean() < 0.05
    b = bin_lightcurve(lc, by="night")
    true_ujy = 10 ** ((23.9 - true_mag) / 2.5)
    pull = (np.asarray(b["flux_ujy"]) - true_ujy) / np.asarray(b["flux_err_ujy"])
    assert abs(np.mean(pull)) < 0.15 and 0.8 < np.std(pull) < 1.2
    assert b["detected"][b["band"] == "g"].mean() > 0.9  # a night of g visits (~10 sigma) detects it


def test_binning_counts_every_measurement_once_and_never_mixes_bands():
    vis = _visits(n=600, seed=7, nights=10)
    lc = inject(vis, _const(17.0), ZP, rng=np.random.default_rng(8))
    for by in ("night", "ratchet", 0.25):
        b = bin_lightcurve(lc, by=by)
        for band in ("g", "rho"):
            assert int(np.sum(b["n"][b["band"] == band])) == int(np.sum(lc["band"] == band))
        assert set(b["band"]) <= {"g", "rho"}
    b = bin_lightcurve(lc, by="ratchet")
    np.testing.assert_allclose(b["mag"], 17.0, atol=5 * np.max(b["mag_err"]))


def test_binned_error_shrinks_as_one_over_root_n():
    vis = _visits(n=240, seed=9, nights=1, slots_per_ratchet=240)
    vis["sigma_bkg_e"] = 40.0
    vis["throughput"] = 0.55
    vis["band"] = "g"
    lc = inject(vis, _const(19.0), ZP, rng=np.random.default_rng(10), floor=0.0)
    b = bin_lightcurve(lc, by="ratchet")
    assert len(b) == 1 and int(b["n"][0]) == 240
    np.testing.assert_allclose(b["flux_err_ujy"][0], lc["flux_err_ujy"][0] / np.sqrt(240), rtol=1e-6)


# --- models that see the exposure --------------------------------------------------------------------------------


def test_a_model_with_a_vis_argument_receives_the_rows_it_is_evaluated_for():
    vis = tbl.vstack([_visits(n=10, seed=11), _visits(n=4, fast=True, seed=12)])

    def model(t, b, vis):
        assert len(vis) == len(t)
        return 15.0 + 0.1 * np.asarray(vis["airmass"])

    out = inject(vis, model, ZP, rng=np.random.default_rng(3))
    np.testing.assert_allclose(out["mag_true"], 15.0 + 0.1 * np.asarray(vis["airmass"]), rtol=1e-9)
    native = inject(vis, model, ZP, rng=np.random.default_rng(3), cadence="native")
    assert len(native) == 10 + 4 * 60


# --- a real run --------------------------------------------------------------------------------------------------

RUN = os.path.expanduser(os.environ.get("ARGUS_LC_RUN", ""))
needs_run = pytest.mark.needs_lc_run


@pytest.fixture(scope="module")
def run(lc_index):
    return SurveyRun(RUN, index_dir=lc_index)


def _a_covered_point(run):
    tid = np.load(os.path.join(RUN, "minipix", "tanseg_id.npy"))
    nobs = np.load(os.path.join(RUN, "minipix", "total_nobs.npy"), mmap_mode="r")
    from argus_sim import skymap_shim as sh

    r = int(np.flatnonzero(np.asarray(nobs[0]).sum(1) > 0)[len(tid) // 7 % 1000])
    ra, dec = sh.tile_center(tid[r : r + 1])
    return float(ra[0]), float(dec[0])


@needs_run
def test_query_filters_equal_filtering_the_full_table(run):
    ra, dec = _a_covered_point(run)
    full = run.visits(ra, dec)
    assert len(full) > 0
    lo, hi = np.percentile(np.asarray(full["mjd"]), [25, 75])
    sub = run.visits(ra, dec, band="g", mjd_min=lo, mjd_max=hi)
    m = (full["band"] == "g") & (full["mjd"] >= lo) & (full["mjd"] <= hi)
    np.testing.assert_array_equal(np.asarray(sub["mjd"]), np.asarray(full["mjd"][m]))
    np.testing.assert_array_equal(np.asarray(sub["tel_id"]), np.asarray(full["tel_id"][m]))
    many = run.visits_many(np.array([ra]), np.array([dec]), band="g", mjd_min=lo, mjd_max=hi)
    assert len(many) == len(sub)


@needs_run
def test_exact_position_is_the_default(run):
    assert inspect.signature(SurveyRun.visits).parameters["position"].default == "exact"
    assert inspect.signature(SurveyRun.visits_many).parameters["position"].default == "exact"


@needs_run
def test_night_log_accounts_for_every_survey_night(run):
    ra, dec = _a_covered_point(run)
    log = run.night_log(ra, dec)
    vis = run.visits(ra, dec)
    assert len(np.unique(log["night"])) == len(log)
    observed = set(np.asarray(log["night"])[np.asarray(log["status"]) == "observed"].tolist())
    assert observed == set(np.unique(np.asarray(vis["night"])).tolist())
    assert set(log["status"]) <= {"observed", "weather", "moon", "not_in_footprint"}
    for row in log[log["status"] == "observed"]:
        assert row["n_slots"] == int(np.sum(np.asarray(vis["night"]) == row["night"]))
    outside = run.night_log(10.0, -70.0)
    assert set(outside["status"]) <= {"weather", "not_in_footprint"}
