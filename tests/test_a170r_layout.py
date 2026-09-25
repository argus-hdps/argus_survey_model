"""The A170r_1200 layout (array_arrangement 9206cc3), the default ring layout, and its band table."""

import hashlib
from importlib.resources import files

import numpy as np
import pytest

from argus_sim.ring_layout import LAYOUTS, load_ring_layout, subarray_members

A170R_SHA256 = "0d2b3154541c5f083e57e8183756696cb234ff8e3e8e660bcf8eb4c045b8c9d1"
BANDS_SHA256 = "51f8b087a78b1f69244a4ebfd334aeb50462eafc21dc6a9959f7755460d4da4c"


@pytest.fixture(scope="module")
def a170r():
    return load_ring_layout("A170r_1200")


def test_shipped_file_is_the_pinned_commit():
    path = files("argus_sim").joinpath("data", "layouts", LAYOUTS["A170r_1200"].filename)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == A170R_SHA256


def test_provenance_names_repo_and_commit():
    text = files("argus_sim").joinpath("data", "layouts", "PROVENANCE.md").read_text()
    assert "argus-hdps/array_arrangement @ 9206cc3b13236a4c9fda2b1164db9e634f7ca241" in text
    assert A170R_SHA256 in text


def test_counts(a170r):
    assert len(a170r) == 1200
    sizes = {s: len(ix) for s, ix in subarray_members(a170r).items()}
    assert sizes == {0: 133, 1: 133, 2: 151, 3: 151, 4: 159, 5: 159, 6: 157, 7: 157}


def test_rings(a170r):
    assert np.unique(a170r["ring_n"]).tolist() == list(range(22))
    np.testing.assert_allclose(a170r["zenith_dist"], 1.0 + a170r["ring_n"] * 2.392494495, atol=1e-4)


def test_each_ring_belongs_to_one_mount_type(a170r):
    """subarray = 2*type + parity; a ring's OTAs are on the two mounts of one type."""
    for r in range(22):
        sub = np.asarray(a170r[a170r["ring_n"] == r]["subarray_n"])
        assert len(set(sub // 2)) == 1


def test_azimuth_offset_and_field():
    assert LAYOUTS["A170r_1200"].az_offset_deg == LAYOUTS["A170r_1200-strips"].az_offset_deg == 90.0
    assert LAYOUTS["A170r_1200"].fov_deg == LAYOUTS["A170r_1200-strips"].fov_deg == (3.252897, 2.442494)


def test_is_the_default_with_its_band_table(a170r):
    from argus_sim.ring_layout import DEFAULT_LAYOUT, load_band_table

    assert DEFAULT_LAYOUT == "A170r_1200"
    assert load_ring_layout().meta["layout"] == "A170r_1200"
    path = files("argus_sim").joinpath("data", "layouts", LAYOUTS["A170r_1200"].band_table)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == BANDS_SHA256
    table = load_band_table(a170r.meta["band_table"])
    assert set(table) == set(int(t) for t in a170r["tel_id"])
    assert sum(v == "b" for v in table.values()) == 600
    assert LAYOUTS["A170r_1200-strips"].band_table is None


def test_band_table_is_mirror_antisymmetric(a170r):
    """Twins (same ring, mirrored about the meridian) carry opposite bands, so every dec line balances."""
    from argus_sim.ring_layout import load_band_table

    table = load_band_table(a170r.meta["band_table"])
    band = {int(t): table[int(t)] for t in a170r["tel_id"]}
    n_pairs = 0
    for r in range(1, 22):
        ring = a170r[a170r["ring_n"] == r]
        az = np.asarray(ring["az"])
        for i, t in enumerate(ring["tel_id"]):
            mirror = (-az[i]) % 360.0
            j = np.flatnonzero(np.abs((az - mirror + 180.0) % 360.0 - 180.0) < 1e-4)
            if len(j) and j[0] != i:
                n_pairs += 1
                assert band[int(t)] != band[int(ring["tel_id"][j[0]])]
    assert n_pairs == 2 * 598
