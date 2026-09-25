import astropy.units as u
import numpy as np
import pytest

from argus_sim.ring_layout import assign_filters, load_ring_layout, ring_pointing


@pytest.fixture(scope="module")
def layout():
    return load_ring_layout("A170r_1200")


@pytest.fixture(scope="module")
def ring_result(layout):
    from argus_sim import c

    return ring_pointing(
        layout,
        c.telescope.long_axis_extent * u.deg,
        c.telescope.short_axis_extent * u.deg,
        c.survey.sim_depth,
    )


class TestLoadRingLayout:
    def test_row_count(self, layout):
        assert len(layout) == 1200

    def test_columns(self, layout):
        for col in ("tel_id", "subarray_n", "ring_n", "az_n", "zenith_dist", "az", "rot_deg"):
            assert col in layout.colnames

    def test_subarray_range(self, layout):
        subs = np.array(layout["subarray_n"])
        assert subs.min() >= 0
        assert subs.max() <= 7

    def test_zenith_dist_positive(self, layout):
        zd = np.array(layout["zenith_dist"])
        assert np.all(zd > 0)

    def test_azimuth_range(self, layout):
        az = np.array(layout["az"])
        assert np.all(az >= 0)
        assert np.all(az < 360)


class TestRingPointing:
    def test_returns_four_tuple(self, ring_result):
        assert len(ring_result) == 4

    def test_cam_table_columns(self, ring_result):
        _, _, cam, _ = ring_result
        required = [
            "x_center",
            "y_center",
            "z_center",
            "alt_center",
            "az_center",
            "ra_center",
            "dec_center",
            "subarray_n",
            "rot_deg",
        ]
        for col in required:
            assert col in cam.colnames

    def test_hpx_table_columns(self, ring_result):
        _, _, _, hpx = ring_result
        for col in ("alt", "az", "hpx"):
            assert col in hpx.colnames

    def test_area_positive(self, ring_result):
        _, area, _, _ = ring_result
        assert area.value > 0

    def test_moc_sky_fraction(self, ring_result):
        moc, _, _, _ = ring_result
        assert 0 < moc.sky_fraction < 1


class TestAssignFilters:
    def test_adds_filter_column(self, ring_result):
        _, _, cam, _ = ring_result
        cam = cam.copy()
        cam = assign_filters(cam, ["A", "g", "r", "i"], cell_size=2.4)
        assert "filter" in cam.colnames

    def test_filter_values_from_options(self, ring_result):
        _, _, cam, _ = ring_result
        cam = cam.copy()
        options = ["A", "g", "r", "i"]
        cam = assign_filters(cam, options, cell_size=2.4)
        assert set(cam["filter"]).issubset(set(options))


class TestStripAssignment:
    def test_three_to_one_ratio(self, ring_result):
        """With options=["g","g","g","r+i"], ~75% should be g."""
        _, _, cam, _ = ring_result
        cam = cam.copy()
        cam = assign_filters(cam, ["g", "g", "g", "r+i"], cell_size=2.4)
        filters = np.array(cam["filter"])
        g_frac = np.sum(filters == "g") / len(filters)
        assert 0.65 < g_frac < 0.85

    def test_subarray_uniformity(self, ring_result):
        """Each subarray should be dominated by a single filter."""
        _, _, cam, _ = ring_result
        cam = cam.copy()
        options = ["g", "g", "g", "r+i"]
        cell_size = 2.4
        cam = assign_filters(cam, options, cell_size=cell_size)

        filters = np.array(cam["filter"])
        subarrays = np.array(cam["subarray_n"])
        for sa in np.unique(subarrays):
            mask = subarrays == sa
            _, counts = np.unique(filters[mask], return_counts=True)
            majority_frac = counts.max() / counts.sum()
            assert majority_frac > 0.6

    def test_both_filters_present(self, ring_result):
        _, _, cam, _ = ring_result
        cam = cam.copy()
        cam = assign_filters(cam, ["g", "g", "g", "r+i"], cell_size=2.4)
        assert set(cam["filter"]) == {"g", "r+i"}

    def test_explicit_cell_size(self, ring_result):
        _, _, cam, _ = ring_result
        cam = cam.copy()
        cam = assign_filters(cam, ["g", "r+i"], cell_size=5.0)
        assert set(cam["filter"]) == {"g", "r+i"}
