"""Parity of the HDPS tessellation shim with hdps.skymap at the pinned panoptes commit.

The reference is `git archive d8dc204b packages/hdps_skymap/src`, unpacked anywhere and named by the environment
variable HDPS_SKYMAP_REF (the directory holding tanseg_ids.py).  It is loaded by path so hdps-core is not needed.
Skipped when HDPS_SKYMAP_REF is unset or the tree is absent.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from argus_sim import skymap_shim as sh

REF = Path(os.environ.get("HDPS_SKYMAP_REF", ""))


@pytest.fixture(scope="module")
def ref():
    if not os.environ.get("HDPS_SKYMAP_REF") or not (REF / "tanseg_ids.py").is_file():
        pytest.skip(f"hdps.skymap reference tree at {sh.PANOPTES_COMMIT} not present")
    pkg = types.ModuleType("_hdps_skymap_ref")
    pkg.__path__ = [str(REF)]
    sys.modules["_hdps_skymap_ref"] = pkg
    if importlib.util.find_spec("rtree") is None:  # imported at module level, used only lazily
        stub = types.ModuleType("rtree")
        stub.index = types.SimpleNamespace(Index=None, Property=None)
        sys.modules.setdefault("rtree", stub)
    mods = {}
    for name in ("tanseg_ids", "tiling_scheme"):
        spec = importlib.util.spec_from_file_location(f"_hdps_skymap_ref.{name}", REF / f"{name}.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m
        spec.loader.exec_module(m)
        mods[name] = m
    return mods


def test_parameters_match(ref):
    t = ref["tanseg_ids"]
    S = ref["tiling_scheme"].SegmentationScheme
    assert (t.MAX_RA_SLOTS, t.MAX_DEC_BANDS, t.MINIPIX_GRID_SIZE, t.MINIPIX_PIXEL_SIZE, t.SEGMENT_PIXEL_SIZE) == (
        sh.MAX_RA_SLOTS,
        sh.MAX_DEC_BANDS,
        sh.MINIPIX_GRID_SIZE,
        sh.MINIPIX_PIXEL_SIZE,
        sh.SEGMENT_PIXEL_SIZE,
    )
    assert (S.DEFAULT_TILE_SIZE_ARCSEC, S.DEFAULT_OVERLAP_ARCSEC, S.DEFAULT_PIXEL_SCALE) == (
        sh.TILE_SIZE_ARCSEC,
        sh.OVERLAP_ARCSEC,
        sh.PIXEL_SCALE_ARCSEC,
    )
    assert sh.SPACING_DEG == 0.4921875


def test_tile_enumeration_matches_generate_tiles(ref):
    S = ref["tiling_scheme"].SegmentationScheme
    scheme = S(
        tile_size_arcsec=sh.TILE_SIZE_ARCSEC, overlap_arcsec=sh.OVERLAP_ARCSEC, pixel_scale=sh.PIXEL_SCALE_ARCSEC
    )
    tiles, _ = scheme.generate_tiles()
    mine = sh.tiles()
    assert len(tiles) == len(mine["tanseg_id"])
    np.testing.assert_array_equal([t["tile_id"] for t in tiles], mine["tanseg_id"])
    np.testing.assert_allclose([t["ra_center"] for t in tiles], mine["ra_center"], rtol=0, atol=1e-12)
    np.testing.assert_allclose([t["dec_center"] for t in tiles], mine["dec_center"], rtol=0, atol=1e-12)


def test_lookup_matches(ref):
    t = ref["tanseg_ids"]
    rng = np.random.default_rng(7)
    n = 200_000
    ra = rng.uniform(0, 360, n)
    dec = np.degrees(np.arcsin(rng.uniform(-0.9999, 0.9999, n)))
    ra = np.r_[ra, [0.0, 359.9999999, 180.0, 279.23]]
    dec = np.r_[dec, [0.0, 45.0, -89.2, 38.78]]
    a = t.radec_to_composite_id(ra, dec, sh.SPACING_DEG, sh.PIXEL_SCALE_DEG)
    b = sh.radec_to_composite_id(ra, dec)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_minipix_centres_match_reverse_lookup(ref):
    t = ref["tanseg_ids"]
    tl = sh.tiles(-30, 85)
    rng = np.random.default_rng(8)
    pick = rng.choice(len(tl["tanseg_id"]), 50, replace=False)
    xi, eta = sh.minipix_offsets()
    for k in pick:
        tid = np.full(sh.N_MINIPIX, tl["tanseg_id"][k])
        ra_ref, dec_ref = t.composite_id_to_radec(tid, np.arange(sh.N_MINIPIX), sh.SPACING_DEG, sh.PIXEL_SCALE_DEG)
        T, E, N = sh.tangent_basis(tl["ra_center"][k], tl["dec_center"][k])
        P = T[None, :] + xi[:, None] * E[None, :] + eta[:, None] * N[None, :]
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        v = np.stack(
            [
                np.cos(np.radians(dec_ref)) * np.cos(np.radians(ra_ref)),
                np.cos(np.radians(dec_ref)) * np.sin(np.radians(ra_ref)),
                np.sin(np.radians(dec_ref)),
            ],
            -1,
        )
        sep = np.degrees(np.arccos(np.clip((P * v).sum(1), -1, 1))) * 3600
        assert sep.max() < 0.01  # arcsec; the reference arcsin loses ~5 mas near high dec
        # forward lookup of the centres agrees with hdps.skymap (centres in a tile's overlap strip belong to the
        # neighbouring tile under floor-strip ownership, in both)
        a_ref, b_ref = t.radec_to_composite_id(ra_ref, dec_ref, sh.SPACING_DEG, sh.PIXEL_SCALE_DEG)
        a, b = sh.radec_to_composite_id(ra_ref, dec_ref)
        assert (a == a_ref).all() and (b == b_ref).all()


def test_backend_parity():
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1).sum()
    except Exception:
        pytest.skip("no working GPU")
    rng = np.random.default_rng(9)
    ra, dec = rng.uniform(0, 360, 100_000), rng.uniform(-80, 85, 100_000)
    a = sh.radec_to_composite_id(ra, dec)
    b = sh.radec_to_composite_id(cp.asarray(ra), cp.asarray(dec), xp=cp)
    np.testing.assert_array_equal(a[0], b[0].get())
    np.testing.assert_array_equal(a[1], b[1].get())
