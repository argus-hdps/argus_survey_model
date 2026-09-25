"""The survey driver: argument handling, configuration, the per-night and span summaries, local scratch."""

import copy
import datetime as dt
import json
import os

import numpy as np
import pytest

from argus_sim import c
from argus_sim import skymap_shim as sh
from argus_sim import survey_driver as drv


@pytest.fixture(autouse=True)
def restore_config():
    saved = {k: copy.deepcopy(getattr(c, k)) for k in ("survey", "output", "filter_strategy", "packing_strategy")}
    nproc = c.nproc
    yield
    for k, v in saved.items():
        setattr(c, k, v)
    c.nproc = nproc


def test_resolve_span():
    t0, t1 = drv.resolve_span("2026-06-14T12:00", n_nights=7)
    assert (t0, t1) == (dt.datetime(2026, 6, 14, 12), dt.datetime(2026, 6, 21, 12))
    assert drv.resolve_span("2026-06-14T12:00", end="2026-06-15T12:00")[1] == dt.datetime(2026, 6, 15, 12)
    for bad in (dict(), dict(end="2026-06-15", n_nights=1), dict(n_nights=0), dict(end="2026-06-14T11:00")):
        with pytest.raises(ValueError):
            drv.resolve_span("2026-06-14T12:00", **bad)


def test_parser_requires_one_span_and_outdir():
    p = drv.build_parser()
    a = p.parse_args(["--start", "2026-06-14T12:00", "--n-nights", "7", "--outdir", "o"])
    assert a.n_nights == 7 and a.end is None and a.local_scratch is None and a.coverage is None
    for argv in (
        ["--start", "2026-06-14T12:00", "--outdir", "o"],
        ["--start", "x", "--n-nights", "1", "--end", "y", "--outdir", "o"],
        ["--start", "2026-06-14T12:00", "--n-nights", "1"],
    ):
        with pytest.raises(SystemExit):
            p.parse_args(argv)


def test_configure_defaults_draw_and_record_a_seed(tmp_path):
    b = drv.build_parser().parse_args(["--start", "2026-06-14T12:00", "--n-nights", "1", "--outdir", "o"])
    s2 = drv.configure(b, str(tmp_path))
    assert isinstance(s2["seed"], int) and s2["seed"] >= 0 and s2["coverage"] == "exact"


def test_configure_applies_and_records(tmp_path):
    a = drv.build_parser().parse_args(
        [
            "--start",
            "2026-06-14T12:00",
            "--n-nights",
            "2",
            "--outdir",
            "o",
            "--seed",
            "1",
            "--cycle",
            "g,g,rho",
            "--coverage",
            "moc",
            "--nproc",
            "2",
        ]
    )
    s = drv.configure(a, str(tmp_path))
    assert c.survey.start_date == dt.datetime(2026, 6, 14, 12) and c.survey.end_date == dt.datetime(2026, 6, 16, 12)
    assert c.survey.coverage == "moc" and c.filter_strategy.options == ["g", "g", "rho"] and c.nproc == 2
    assert c.output.output_dir == str(tmp_path)
    assert s["seed"] == 1 and s["ratchet_len"] == c.survey.ratchet_len and s["reset_min"] == c.survey.reset_min


def _manifest(nights=3):
    rows = []
    for k in range(nights):
        nt = {
            "night": 61205 + k,
            "mjd_min": 61206.15 + k,
            "mjd_max": 61206.42 + k,
            "n_ratchets": 27,
            "moon_frac_min": 0.01 * k,
            "moon_frac_median": 0.02 * k,
            "moon_frac_max": 0.8 * k,
            "n_fast_ratchets": 3 * k,
        }
        for band, depth in (("g", 24.0), ("rho", 23.0)):
            if k == 1 and band == "rho":
                continue  # a night with no rho coverage is skipped for that band
            nt.update(
                {
                    f"{band}_n_minipix": 100 + k,
                    f"{band}_area_deg2": 1.0 + k,
                    f"{band}_median_limmag": depth + 0.01 * k,
                    f"{band}_p16_limmag": depth - 0.5,
                    f"{band}_p84_limmag": depth + 0.4,
                    f"{band}_median_n_obs": 112.0,
                    f"{band}_mean_n_obs": 113.0,
                }
            )
        rows.append(nt)
    return {
        "coverage": "exact",
        "bands": ["g", "rho"],
        "nights": rows,
        "band_zeropoints": {
            "g": {"zp_photons_per_sec": 1e10, "band_qe": 0.8},
            "rho": {"zp_photons_per_sec": 2e10, "band_qe": 0.7},
        },
    }


def test_night_table_has_every_night():
    rows = drv.night_table(_manifest(3))
    assert [(r["night"], r["band"]) for r in rows] == [
        (61205, "g"),
        (61205, "rho"),
        (61206, "g"),
        (61207, "g"),
        (61207, "rho"),
    ]
    assert rows[2]["median_limmag"] == pytest.approx(24.01)
    assert rows[3]["n_fast_ratchets"] == 6 and rows[3]["moon_frac_max"] == pytest.approx(1.6)


def _write_totals(d, tile_ids):
    m = os.path.join(d, "minipix")
    os.makedirs(m, exist_ok=True)
    n = len(tile_ids)
    np.save(os.path.join(m, "tanseg_id.npy"), tile_ids)
    ivar = np.zeros((2, n, sh.N_MINIPIX), np.float32)
    nobs = np.zeros((2, n, sh.N_MINIPIX), np.uint32)
    ivar[0, 0], nobs[0, 0] = 1e-4, 100
    ivar[1, 1], nobs[1, 1] = 3e-5, 50
    np.save(os.path.join(m, "total_ivar.npy"), ivar)
    np.save(os.path.join(m, "total_nobs.npy"), nobs)


def test_span_summary_owned_minipixes(tmp_path):
    tl = sh.tiles(10.0, 10.6)
    tid = tl["tanseg_id"][:2]
    _write_totals(str(tmp_path), tid)
    out = drv.span_summary(str(tmp_path), _manifest())
    from argus_sim.tanseg_coverage import owned_mask

    own = owned_mask(tid)
    assert out["g"]["n_minipix"] == int(own[0].sum()) and out["rho"]["n_minipix"] == int(own[1].sum())
    rate = c.survey.detection_snr / np.sqrt(np.float32(1e-4)) / c.survey.base_cadence_s / 0.8
    assert out["g"]["median_limmag"] == pytest.approx(-2.5 * np.log10(rate / 1e10), abs=1e-5)
    assert out["g"]["median_n_obs"] == 100 and out["rho"]["median_n_obs"] == 50


def test_run_with_local_scratch_copies_and_summarises(tmp_path, monkeypatch):
    tl = sh.tiles(10.0, 10.6)

    class FakeSurvey:
        def run(self, seed=None):
            _write_totals(c.output.output_dir, tl["tanseg_id"][:2])
            os.makedirs(os.path.join(c.output.output_dir, "epochs"), exist_ok=True)
            return _manifest(3)

    import argus_sim.survey as survey_mod

    monkeypatch.setattr(survey_mod, "Survey", FakeSurvey)
    out, local = tmp_path / "final", tmp_path / "local"
    s = drv.run(
        [
            "--start",
            "2026-06-14T12:00",
            "--n-nights",
            "3",
            "--outdir",
            str(out),
            "--local-scratch",
            str(local),
            "--seed",
            "1",
            "--no-record-stats",
        ]
    )
    assert os.path.isfile(out / "night_summary.csv") and os.path.isfile(out / "span_summary.json")
    assert list(local.iterdir()) == []  # the local working copy is removed after the copy
    prov = json.loads((out / "provenance.json").read_text())
    for key in ("git_hash", "coverage", "ring_layout", "band_assignment", "ratchet_len", "reset_min", "seed"):
        assert key in prov
    assert prov["seed"] == 1 and prov["workdir"] != prov["outdir"]
    assert len(s["nights"]) == 5 and set(s["span"]) == {"g", "rho"}
    with open(out / "night_summary.csv") as f:
        assert len(f.read().strip().splitlines()) == 6
