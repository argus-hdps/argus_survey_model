"""argus_sim.lightcurve: a compact, self-contained store readable from S3.

Uses the survey run named by ARGUS_LC_RUN (see test_lightcurve.py) and skips without it. The S3 test needs moto
(dev dependency) and skips without it.
"""

import json
import os
import shutil

import numpy as np
import pytest

from argus_sim.lightcurve import SurveyRun

RUN = os.path.expanduser(os.environ.get("ARGUS_LC_RUN", ""))
needs_run = pytest.mark.needs_lc_run
COLUMNS = (
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
)


@pytest.fixture(scope="module")
def store(lc_index, tmp_path_factory):
    """The run's compact store, copied alone to a directory with no run beside it."""
    alone = str(tmp_path_factory.mktemp("alone") / "store")
    shutil.copytree(lc_index, alone, ignore=shutil.ignore_patterns("build"))
    return alone


@pytest.fixture(scope="module")
def run(store):
    return SurveyRun(store)


def _points(run, n, seed=0):
    """Tile centres of tiles the run covered."""
    from argus_sim import skymap_shim as sh

    tid = np.load(os.path.join(RUN, "minipix", "tanseg_id.npy"))
    nobs = np.load(os.path.join(RUN, "minipix", "total_nobs.npy"), mmap_mode="r")
    rows = np.flatnonzero(np.asarray(nobs[0]).sum(1) > 0)
    rows = np.random.default_rng(seed).choice(rows, size=n, replace=False)
    ra, dec = sh.tile_center(tid[rows])
    return list(zip(np.asarray(ra, float), np.asarray(dec, float)))


def _survey_records(ra, dec):
    """The run's own per-pair values at a tile centre, read straight from the epoch records (full precision)."""
    from argus_sim import skymap_shim as sh

    t, _ = sh.radec_to_composite_id(np.array([ra]), np.array([dec]))
    out = {}
    for f in sorted(os.listdir(os.path.join(RUN, "epochs"))):
        d = np.load(os.path.join(RUN, "epochs", f))
        sel = np.flatnonzero(d["tanseg_id"] == t[0])
        for i in sel:
            out[(int(d["ratchet"]), int(d["tel_id"][i]))] = dict(
                sigma_bkg_e=float(d["noise"][i]),
                throughput=float(d["signal_tp"][i]),
                airmass=float(d["airmass"][i]),
                field_angle=float(d["field_angle"][i]),
            )
    return out


# --- self-contained ------------------------------------------------------------------------------------------------


@needs_run
def test_store_opens_with_nothing_beside_it(store, run):
    assert not os.path.exists(os.path.join(os.path.dirname(store), "epochs"))
    assert not os.path.exists(os.path.join(store, "build"))
    man = json.load(open(os.path.join(RUN, "exact_manifest.json")))
    assert run.zeropoints == man["band_zeropoints"]
    idx = json.load(open(os.path.join(store, "index.json")))
    for key in ("format_version", "zeropoints", "plan", "quantisation", "argus_sim_commit", "layout", "span"):
        assert key in idx, key


@needs_run
def test_visits_have_every_column_and_match_the_run_records(run):
    """Recomputed geometry and 16-bit noise columns reproduce the run's own per-pair values."""
    for ra, dec in _points(run, 5):
        vis = run.visits(ra, dec, position="minipix")
        assert len(vis) > 0
        for col in COLUMNS:
            assert col in vis.colnames, col
        truth = _survey_records(ra, dec)
        for row in vis:
            t = truth[(int(row["ratchet"]), int(row["tel_id"]))]
            assert abs(row["sigma_bkg_e"] / t["sigma_bkg_e"] - 1) <= 1e-4
            assert abs(row["throughput"] / t["throughput"] - 1) <= 1e-4
            assert abs(row["airmass"] - t["airmass"]) <= 1e-4 * t["airmass"]
            assert abs(row["field_angle"] - t["field_angle"]) <= 1e-3  # deg


@needs_run
def test_store_is_compact(store):
    idx = json.load(open(os.path.join(store, "index.json")))
    pair_bytes = sum(os.path.getsize(os.path.join(store, "pairs", f)) for f in os.listdir(os.path.join(store, "pairs")))
    assert pair_bytes / idx["n_pair_rows"] <= 6.0


# --- summaries ---------------------------------------------------------------------------------------------------


@needs_run
def test_nights_table_is_the_run_night_summary(run):
    import pandas as pd

    ref = pd.read_csv(os.path.join(RUN, "night_summary.csv"))
    got = run.nights().to_pandas()
    assert len(got) == len(ref)
    np.testing.assert_allclose(np.sort(got["median_limmag"]), np.sort(ref["median_limmag"]), rtol=1e-6)


@needs_run
def test_depth_maps_cover_every_period_and_add_up(run):
    periods = run.periods
    assert periods[-1] == "all" and len(periods) >= 2
    for band in run.bands:
        total = run.depth_map(band, "all")
        for col in ("tanseg_id", "ra", "dec", "limmag", "n_nights", "n_slots_60", "n_slots_1", "n_frames"):
            assert col in total.colnames, col
        parts = [run.depth_map(band, p) for p in periods[:-1]]
        frames = {int(t): 0 for t in total["tanseg_id"]}
        for part in parts:
            for t, n in zip(part["tanseg_id"], part["n_frames"]):
                frames[int(t)] += int(n)
        assert all(frames[int(t)] == int(n) for t, n in zip(total["tanseg_id"], total["n_frames"]))


@needs_run
def test_tile_depth_agrees_with_the_minipix_totals(run):
    """A tile's 5-year depth is within 0.1 mag of the median over its owned minipixes (fully covered tiles)."""
    from argus_sim.tanseg_coverage import FAST_UNIT, owned_mask

    man = json.load(open(os.path.join(RUN, "exact_manifest.json")))
    zp = man["band_zeropoints"]
    tid = np.load(os.path.join(RUN, "minipix", "tanseg_id.npy"))
    ivar = np.load(os.path.join(RUN, "minipix", "total_ivar.npy"), mmap_mode="r")
    nobs = np.load(os.path.join(RUN, "minipix", "total_nobs.npy"), mmap_mode="r")
    fpath = os.path.join(RUN, "minipix", "total_fast_units.npy")
    fast = np.load(fpath, mmap_mode="r") if os.path.isfile(fpath) else None
    for b, band in enumerate(run.bands):
        dm = run.depth_map(band, "all")
        row_of = {int(t): i for i, t in enumerate(tid)}
        full = dm[dm["covered_fraction"] > 0.99]
        pick = np.random.default_rng(3).choice(len(full), size=min(50, len(full)), replace=False)
        for i in pick:
            r = row_of[int(full["tanseg_id"][i])]
            own = np.asarray(owned_mask(tid[r : r + 1]))[0]
            iv = np.asarray(ivar[b, r], float)[own]
            n60 = np.asarray(nobs[b, r], float)[own]
            n1 = np.asarray(fast[b, r], float)[own] * FAST_UNIT if fast is not None else 0 * n60
            ok = iv > 0
            rate = 5 / np.sqrt(iv[ok]) / ((n60[ok] * 60 + n1[ok]) / (n60[ok] + n1[ok])) / zp[band]["band_qe"]
            ref = np.median(-2.5 * np.log10(rate / zp[band]["zp_photons_per_sec"]))
            assert abs(float(full["limmag"][i]) - ref) < 0.1


# --- S3 ----------------------------------------------------------------------------------------------------------


@needs_run
def test_the_same_store_on_s3_gives_the_same_visits_in_one_request_per_query(store, run):
    moto_server = pytest.importorskip("moto.server")
    import boto3

    server = moto_server.ThreadedMotoServer(port=0, verbose=False)
    server.start()
    try:
        host, port = server.get_host_and_port()
        endpoint = f"http://{host}:{port}"
        s3 = boto3.client(
            "s3", endpoint_url=endpoint, aws_access_key_id="x", aws_secret_access_key="x", region_name="us-east-1"
        )
        s3.create_bucket(Bucket="argus")
        for root, _, files in os.walk(store):
            for f in files:
                p = os.path.join(root, f)
                s3.upload_file(p, "argus", "store/" + os.path.relpath(p, store))
        opts = dict(
            endpoint_override=f"{host}:{port}", scheme="http", access_key="x", secret_key="x", region="us-east-1"
        )
        remote = SurveyRun("s3://argus/store", storage_options=opts)
        pts = _points(run, 4, seed=5)
        for ra, dec in pts:
            a, b = run.visits(ra, dec), remote.visits(ra, dec)
            np.testing.assert_array_equal(np.asarray(a["mjd"]), np.asarray(b["mjd"]))
            np.testing.assert_array_equal(np.asarray(a["sigma_bkg_e"]), np.asarray(b["sigma_bkg_e"]))
        # after the shard has been opened once, a point query is one ranged GET
        ra, dec = pts[0]
        n0 = remote.io_stats()["get_requests"]
        remote.visits(ra, dec)
        assert remote.io_stats()["get_requests"] - n0 <= 1
    finally:
        server.stop()
