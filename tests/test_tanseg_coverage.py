"""Exact coverage on the HDPS tanseg/minipix grid: closure on the layout's figures, the southern edge,
parity with coverage_exact.covering() on minipix centres, backend parity, the accumulator and the depth formula."""

import astropy.time as atime
import erfa
import numpy as np
import pytest

from argus_sim import c
from argus_sim import skymap_shim as sh
from argus_sim.coverage_exact import Layout, Schedule, covering, crds, get_backend, u
from argus_sim.tanseg_coverage import (
    MINIPIX_AREA_DEG2,
    MinipixAccumulator,
    RatchetPairs,
    TansegCoverage,
    icrs_to_array_rotation,
    minipix_limmag,
    owned_mask,
)

NIGHT = (atime.Time("2026-06-15T00:00"), atime.Time("2026-06-16T00:00"))


@pytest.fixture(scope="module")
def layout():
    return Layout.from_config("A170r_1200", ["g", "rho", "g", "rho"])


@pytest.fixture(scope="module")
def ratchet():
    site = crds.EarthLocation(lat=c.observatory.latitude * u.deg, lon=c.observatory.longitude * u.deg)
    ex = Schedule().exposures(*NIGHT, site)
    first = np.unique(ex["ratchet"], return_index=True)[1]
    i = first[10]
    return ex["time"][i], float(ex["theta_deg"][i]), ex


@pytest.fixture(scope="module")
def engine(layout):
    return TansegCoverage(layout, backend="auto")


@pytest.fixture(scope="module")
def one(engine, ratchet):
    t, th, _ = ratchet
    return engine.pairs(icrs_to_array_rotation(t, th))


def _h(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _looks(engine, p):
    """Covering OTA count per (tile row, minipix), over one ratchet, as a dict of dense rows."""
    xp = engine.xp
    rows = xp.unique(xp.concatenate([p.full_row, p.part_row]))
    cnt = xp.zeros((len(rows), sh.N_MINIPIX), xp.int32)
    xp.add.at(cnt, xp.searchsorted(rows, p.full_row), 1)
    xp.add.at(cnt, xp.searchsorted(rows, p.part_row), p.part_mask.astype(xp.int32))
    return rows, cnt


def test_closes_on_the_layout_figures(engine, one):
    """Owned covered minipixes: 8,058 deg2 instantaneous; >= 2 OTAs on 17.6% of the footprint (array_arrangement's figures)."""
    xp = engine.xp
    rows, cnt = _looks(engine, one)
    own = owned_mask(engine.tanseg_id[_h(rows)], xp)
    covered = (cnt > 0) & own
    area = float(covered.sum()) * MINIPIX_AREA_DEG2
    assert area == pytest.approx(8058, rel=0.004)
    assert float(((cnt >= 2) & own).sum()) / float(covered.sum()) == pytest.approx(0.176, abs=0.004)


def test_southern_edge_is_the_layout_edge(engine, ratchet):
    """The southern edge of the footprint at the default site is the layout edge (of-date declination -21.47 deg),
    not the edge of NSIDE-256 cells.

    The edge is a CIRS declination: in ICRS it moves with RA by up to ~0.15 deg in 2026 (precession since J2000),
    so the minipix centres are carried to CIRS before their declination is taken.
    """
    t, th, ex = ratchet
    xp = engine.xp
    xi, eta = sh.minipix_offsets(xp)
    first = np.unique(ex["ratchet"], return_index=True)[1]
    for i in first[[0, 10, 20]]:
        t = ex["time"][i]
        p = engine.pairs(icrs_to_array_rotation(t, float(ex["theta_deg"][i])))
        M = xp.asarray(erfa.c2i06a(t.tt.jd1, t.tt.jd2))
        dmin = 99.0
        for rows, masks in ((p.full_row, None), (p.part_row, p.part_mask)):
            sel = engine.T[rows, 2] < np.sin(np.radians(-20))
            r = rows[sel]
            P = (
                engine.T[r][:, None, :]
                + xi[None, :, None] * engine.E[r][:, None, :]
                + eta[None, :, None] * engine.N[r][:, None, :]
            )
            P /= xp.linalg.norm(P, axis=-1, keepdims=True)
            dec = xp.degrees(xp.arcsin((P @ M.T)[..., 2]))
            if masks is not None:
                dec = xp.where(masks[sel], dec, 99.0)
            if dec.size:
                dmin = min(dmin, float(dec.min()))
        assert dmin == pytest.approx(-21.47, abs=0.01)


def test_matches_covering_on_minipix_centres(engine, one, layout, ratchet):
    t, th, ex = ratchet
    rng = np.random.default_rng(4)
    rows, cnt = _looks(engine, one)
    rows_h, cnt_h = _h(rows), _h(cnt)
    pick = rng.choice(len(rows_h), 300, replace=False)
    mp = rng.integers(0, sh.N_MINIPIX, (300, 40))
    xi, eta = sh.minipix_offsets()
    T, E, N = (_h(a[rows[pick]]) for a in (engine.T, engine.E, engine.N))
    P = T[:, None, :] + xi[mp][..., None] * E[:, None, :] + eta[mp][..., None] * N[:, None, :]
    P /= np.linalg.norm(P, axis=-1, keepdims=True)
    # ICRS -> CIRS with the same bias-precession-nutation the engine applies (no aberration)
    Pc = P.reshape(-1, 3) @ erfa.c2i06a(t.tt.jd1, t.tt.jd2).T
    ra = np.degrees(np.arctan2(Pc[:, 1], Pc[:, 0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(Pc[:, 2], -1, 1)))
    # the one ratchet the engine evaluated
    k = ex["ratchet"][np.argmin(np.abs(ex["time"].mjd - t.mjd))]
    sel = ex["ratchet"] == k
    t0, t1 = ex["time"][sel][0] - 1 * u.s, ex["time"][sel][-1] + 1 * u.s
    cv = covering(ra, dec, t0, t1, layout=layout, backend="numpy")
    n_exp = sel.sum()
    ref = np.bincount(cv.point, minlength=len(ra)) // n_exp
    mine = cnt_h[pick[:, None], mp].ravel()
    assert (ref == mine).all()


def test_backends_agree(layout, ratchet):
    if get_backend("auto") is np:
        pytest.skip("no working GPU")
    t, th, _ = ratchet
    R = icrs_to_array_rotation(t, th)
    a = TansegCoverage(layout, backend="numpy", dec_range=(20.0, 40.0)).pairs(R)
    b = TansegCoverage(layout, backend="cupy", dec_range=(20.0, 40.0)).pairs(R)
    for k in ("full_row", "full_ota", "part_row", "part_ota", "part_mask"):
        assert np.array_equal(getattr(a, k), _h(getattr(b, k)))


def test_same_band_looks_add(layout):
    eng = TansegCoverage(layout, backend="numpy", dec_range=(30.0, 31.0))
    acc = MinipixAccumulator(eng, None)
    g_otas = np.flatnonzero(eng.ota_band == 0)[:2]
    mask = np.zeros((1, sh.N_MINIPIX), bool)
    mask[0, :100] = True
    p = RatchetPairs(
        full_row=np.array([3, 3]),
        full_ota=g_otas,
        part_row=np.array([5]),
        part_ota=g_otas[:1],
        part_mask=mask,
        n_tiles_in_view=0,
    )
    w = np.zeros(eng.n_tiles)
    w[[3, 5]] = 2.0
    acc.add(p, {0: w, 1: w}, n_epochs=14)
    assert (acc.nobs[0, 3] == 28).all() and np.allclose(acc.ivar[0, 3], 56.0)
    assert (acc.nobs[0, 5, :100] == 14).all() and (acc.nobs[0, 5, 100:] == 0).all()
    assert acc.nobs[1].sum() == 0


def test_depth_formula_matches_ab(layout):
    from argus_sim import ABPhot, SystemThroughput

    tp = SystemThroughput(throughput_loss=c.telescope.throughput_loss, filters={"g": True, "rho": True})
    ab = ABPhot(collecting_area=c.telescope.collecting_area, throughput_loss=c.telescope.throughput_loss, throughput=tp)
    ivar = np.array([[1e-4, 3e-5]])
    nobs = np.array([[100, 50]], np.uint16)
    m = minipix_limmag(ivar, nobs, None, ab, "g", 0.8, 5.0, 60.0, 1.0)
    zp = float(ab.photons_from_mag("g", 0.0).value)
    rate = 5.0 / np.sqrt(ivar) / 60.0 / 0.8
    np.testing.assert_allclose(m, -2.5 * np.log10(rate / zp), atol=1e-9)


def test_fast_exposures_counted_in_units_and_chunking_is_exact(layout):
    from argus_sim.tanseg_coverage import FAST_UNIT

    eng = TansegCoverage(layout, backend="numpy", dec_range=(30.0, 32.0))
    g_otas = np.flatnonzero(eng.ota_band == 0)[:2]
    rng = np.random.default_rng(3)
    rows = np.sort(rng.choice(eng.n_tiles, 40, replace=False))
    mask = rng.random((20, sh.N_MINIPIX)) < 0.4
    p = RatchetPairs(
        full_row=rows[:20],
        full_ota=np.resize(g_otas, 20),
        part_row=rows[20:],
        part_ota=np.resize(g_otas, 20),
        part_mask=mask,
        n_tiles_in_view=0,
    )
    w = np.zeros(eng.n_tiles)
    w[rows] = rng.random(40)
    a = MinipixAccumulator(eng, None, chunk_rows=4096)
    b = MinipixAccumulator(eng, None, chunk_rows=7)  # many chunks, same answer
    for acc in (a, b):
        acc.add(p, {0: w, 1: w}, n_epochs=14)
        acc.add(p, {0: w, 1: w}, n_epochs=14 * FAST_UNIT, fast=True)
    assert np.array_equal(a.nobs, b.nobs) and np.array_equal(a.fast_units, b.fast_units)
    np.testing.assert_allclose(a.ivar, b.ivar, rtol=1e-6)
    assert (a.nobs[0, rows[0]] == 14).all() and (a.fast_units[0, rows[0]] == 14).all()
    with pytest.raises(ValueError):
        a.add(p, {0: w, 1: w}, n_epochs=100, fast=True)


def test_worker_returns_full_system_throughput():
    import astropy.table as tbl
    import astropy.time as atime2

    from argus_sim.observatory import pickering_airmass
    from argus_sim.spectral_sky import build_band_params
    from argus_sim.survey import Survey

    from argus_sim.spectral_sky import signal_throughput

    sv = Survey(with_cradle=False)
    bands = sorted(set(c.filter_strategy.options))
    sv._sharpness_tables = {b: (np.array([0.0]), np.array([0.5, 4.0]), np.full((1, 2), 0.05)) for b in bands}
    bp = build_band_params(bands, sv.throughputs, sv.ab)
    alt = np.array([80.0, 55.0, 40.0])
    grid = tbl.Table({"alt": alt, "az": np.array([10.0, 120.0, 250.0]), "filter": np.array([bands[0]] * 3)})
    res = sv._proc_ratchet(
        0, atime2.Time("2026-06-15T06:00:00"), bp, 0.87, 0.15, None, None, 1, n_dark=14, grid_tab=grid
    )
    exp = signal_throughput(bp[bands[0]], pickering_airmass(alt)) * res["transparency"]
    np.testing.assert_allclose(res["signal_tp"], exp, rtol=1e-12)
