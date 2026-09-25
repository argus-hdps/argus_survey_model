"""The canonical layout as CradlePointing uses it: its camera field, subarrays and band assignment."""

import astropy.units as u
import numpy as np
import pytest

from argus_sim import c
from argus_sim.ring_layout import DEFAULT_LAYOUT, load_ring_layout, ring_pointing


@pytest.fixture(scope="module")
def canonical():
    return load_ring_layout(DEFAULT_LAYOUT)


def test_fov(canonical):
    assert canonical.meta["fov_deg"] == (3.252897, 2.442494)


def test_ring_pointing_carries_structure(canonical):
    _, area, cam, _ = ring_pointing(canonical, 3.252897 * u.deg, 2.442494 * u.deg, c.survey.sim_depth)
    for col in ("tel_id", "subarray_n", "ring_n", "az_n"):
        assert col in cam.colnames
    assert cam.meta["layout"] == "A170r_1200"
    # MOC union at sim_depth includes partially covered cells, so it sits a little above 8058 deg2
    assert 8000 < area.to_value(u.deg**2) < 8200


def test_cradle_uses_layout_fov_and_exposes_subarrays():
    from argus_sim.grid import CradlePointing

    cp = CradlePointing()
    assert cp.layout == "A170r_1200"
    assert cp.long_axis_extent.to_value(u.deg) == pytest.approx(3.252897)
    assert cp.short_axis_extent.to_value(u.deg) == pytest.approx(2.442494)
    assert sum(len(ix) for ix in cp.subarrays.values()) == 1200
    assert set(cp.subarrays) == set(range(8))


def test_explicit_extents_win():
    from argus_sim.grid import CradlePointing

    cp = CradlePointing(long_axis_extent=3.3 * u.deg, short_axis_extent=2.5 * u.deg)
    assert cp.long_axis_extent.to_value(u.deg) == pytest.approx(3.3)


def test_table_applies_only_to_equal_two_band_cycles():
    from argus_sim.ring_layout import table_applies

    assert table_applies(["g", "rho", "g", "rho"])
    assert table_applies(["rho", "g"])
    assert not table_applies(["g", "g", "rho"])
    assert not table_applies(["g", "r", "i", "g"])


def _cradle(layout, options):
    from argus_sim.grid import CradlePointing

    old = (c.packing_strategy.layout_file, list(c.filter_strategy.options))
    c.packing_strategy.layout_file, c.filter_strategy.options = layout, options
    try:
        return CradlePointing()
    finally:
        c.packing_strategy.layout_file, c.filter_strategy.options = old


def test_default_uses_its_band_table():
    cp = _cradle(None, ["g", "rho", "g", "rho"])
    f = np.asarray(cp.cam_pointings["filter"])
    assert cp.band_assignment == "table:A170r_1200_bands_alt5_2026-09-23.csv"
    assert ((f == "g").sum(), (f == "rho").sum()) == (600, 600)


def test_strip_rule_still_selectable():
    cp = _cradle("A170r_1200-strips", ["g", "rho", "g", "rho"])
    f = np.asarray(cp.cam_pointings["filter"])
    assert cp.band_assignment.startswith("strips:")
    n_g, n_rho = (f == "g").sum(), (f == "rho").sum()
    assert n_g + n_rho == 1200
    assert 0.45 < n_g / 1200 < 0.55


def test_two_to_one_falls_back_to_strips():
    cp = _cradle(None, ["g", "g", "rho"])
    assert cp.band_assignment.startswith("strips:")


def test_provenance_names_band_assignment():
    from argus_sim.provenance import layout_state

    old = list(c.filter_strategy.options)
    try:
        c.filter_strategy.options = ["g", "rho", "g", "rho"]
        assert layout_state()["band_assignment"] == "table:A170r_1200_bands_alt5_2026-09-23.csv"
        c.filter_strategy.options = ["g", "g", "rho"]
        assert layout_state()["band_assignment"] == "strips"
    finally:
        c.filter_strategy.options = old
