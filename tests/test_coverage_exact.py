"""Exact per-point coverage: closure against the layout's figures from array_arrangement, an independent
brute-force test, backend parity and the ratchet schedule."""

import astropy.time as atime
import astropy_healpix as ahpx
import numpy as np
import pytest

from argus_sim import coverage_exact as ce


@pytest.fixture(scope="module")
def layout():
    return ce.Layout.from_config("A170r_1200", ["g", "rho", "g", "rho"])


@pytest.fixture(scope="module")
def sky():
    rng = np.random.default_rng(1)
    n = 2000
    dec = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(-22)), np.sin(np.radians(83.5)), n)))
    return rng.uniform(0, 360, n), dec


NIGHT = (atime.Time("2026-06-15T00:00"), atime.Time("2026-06-16T00:00"))


def _brute(layout, alt, az):
    """The gnomonic membership test written out directly, every point against every OTA."""
    C, et, er = layout.basis()
    P = ce._unit(alt, az, np)
    d = P @ C.T
    tl, ts = np.tan(np.radians(layout.fov_deg[0] / 2)), np.tan(np.radians(layout.fov_deg[1] / 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (d > 0) & (np.abs((P @ et.T) / d) < tl) & (np.abs((P @ er.T) / d) < ts)


def test_candidate_table_misses_nothing(layout):
    rng = np.random.default_rng(3)
    zd = np.degrees(np.arccos(rng.uniform(np.cos(np.radians(54)), 1, 5000)))
    az = rng.uniform(0, 360, 5000)
    ref = _brute(layout, 90 - zd, az)
    rows, otas = ce._members(layout, 90 - zd, az, np)
    got = np.zeros_like(ref)
    got[rows, otas] = True
    assert (got == ref).all()


def test_closes_on_the_layout_figures(layout):
    """array_arrangement's figures for A170r_1200: 8,058 deg2; within zd 52, 17.6% covered by >= 2 OTAs, none
    uncovered."""
    hp = ahpx.HEALPix(nside=1024, order="ring")
    lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))
    keep = lat.deg > 90 - 56
    alt, az = lat.deg[keep], lon.deg[keep]
    rows, _ = ce._members(layout, alt, az, np)
    n = np.bincount(rows, minlength=len(alt))
    area = (n > 0).sum() * hp.pixel_area.to_value("deg2")
    assert area == pytest.approx(8058, rel=0.002)
    inner = (90 - alt) < 52
    assert np.mean(n[inner] == 1) == pytest.approx(0.824, abs=0.003)
    assert np.mean(n[inner] >= 2) == pytest.approx(0.176, abs=0.003)
    assert np.mean(n[inner] == 0) < 1e-4


def test_backends_agree(layout, sky):
    pytest.importorskip("cupy")
    if ce.get_backend("auto") is np:
        pytest.skip("no working GPU")
    a = ce.covering(*sky, *NIGHT, layout=layout, backend="numpy")
    b = ce.covering(*sky, *NIGHT, layout=layout, backend="cupy")
    assert b.backend == "cupy"
    for k in ("point", "exposure", "tel_id", "subarray", "band"):
        assert (getattr(a, k) == getattr(b, k)).all()


def test_schedule_is_the_ratchet(layout):
    sch = ce.Schedule()
    from argus_sim.coverage_exact import crds, u
    from argus_sim import c

    site = crds.EarthLocation(lat=c.observatory.latitude * u.deg, lon=c.observatory.longitude * u.deg)
    ex = sch.exposures(*NIGHT, site)
    per_ratchet = np.bincount(ex["ratchet"] - ex["ratchet"].min())
    assert per_ratchet.max() == 15
    assert ex["theta_deg"].max() < 3.75 and ex["theta_deg"].min() > -0.02  # 15 min at the sidereal rate
    gaps = np.diff(ex["time"].mjd) * 1440
    assert np.isclose(gaps.min(), 1.0) and np.isclose(np.sort(gaps)[-2], 2.0)  # the reset minute
    sun = crds.get_body("sun", ex["time"]).transform_to(crds.AltAz(obstime=ex["time"], location=site)).alt.deg
    assert (sun < -18).all()


def test_tracked_position_keeps_its_otas_within_a_ratchet(layout, sky):
    cv = ce.covering(*sky, *NIGHT, layout=layout, backend="numpy")
    key = cv.point * 10_000_000 + cv.ratchet[cv.exposure] * 1000
    sets = {}
    for k, e, t in zip(key, cv.exposure, cv.tel_id):
        sets.setdefault(k, {}).setdefault(e, set()).add(t)
    varying = sum(len({frozenset(v) for v in d.values()}) > 1 for d in sets.values())
    assert varying == 0  # sidereal-rate tracking holds a position exactly still within a ratchet


def test_refraction_and_aberration_flags(layout, sky):
    assert ce.refraction_deg(45.0) * 60 == pytest.approx(
        1.0127 * 790 / 1010, abs=0.001
    )  # arcmin: Saemundsson at 45 deg, 790 mb, 10 C
    base = ce.covering(*sky, *NIGHT, layout=layout, backend="numpy")
    refr = ce.covering(*sky, *NIGHT, layout=layout, backend="numpy", refraction=True)
    app = ce.covering(*sky, *NIGHT, layout=layout, backend="numpy", apparent=True)
    for other in (refr, app):
        assert len(other.point) == pytest.approx(len(base.point), rel=0.02)
        assert not (len(other.point) == len(base.point) and (other.tel_id == base.tel_id).all())


def test_intervals_are_whole_ratchets(layout, sky):
    iv = ce.coverage_intervals(sky[0][:200], sky[1][:200], *NIGHT, layout=layout, backend="numpy")
    n = np.array([r["n_exposures"] for r in iv])
    assert n.max() == 15
    assert np.mean(n == 15) > 0.9
    assert all(r["t_out"] >= r["t_in"] for r in iv)
