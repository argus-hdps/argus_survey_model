import astropy.units as u
import healsparse as hsp
import numpy as np
import pytest

from argus_sim import c
from argus_sim.ring_layout import assign_filters, load_ring_layout, ring_pointing


@pytest.fixture(scope="module")
def cam_table():
    layout = load_ring_layout("A170r_1200")
    _, _, cam, _ = ring_pointing(
        layout,
        c.telescope.long_axis_extent * u.deg,
        c.telescope.short_axis_extent * u.deg,
        c.survey.sim_depth,
    )
    cell_size = c.filter_strategy.cell_size
    if cell_size is None:
        cell_size = c.telescope.short_axis_extent
    return assign_filters(cam, c.filter_strategy.options, cell_size)


@pytest.fixture(scope="module")
def filter_maps(cam_table):
    from argus_sim.coverage_map import build_filter_maps

    return build_filter_maps(
        cam_table,
        c.telescope.long_axis_extent * u.deg,
        c.telescope.short_axis_extent * u.deg,
        c.survey.sim_depth,
        2**c.survey.sim_depth,
    )


class TestBuildFilterMaps:
    def test_returns_dict_with_filter_keys(self, filter_maps):
        assert isinstance(filter_maps, dict)
        for filt in c.filter_strategy.options:
            assert filt in filter_maps

    def test_maps_are_healsparse(self, filter_maps):
        for hsmap in filter_maps.values():
            assert isinstance(hsmap, hsp.HealSparseMap)

    def test_maps_dtype_int16(self, filter_maps):
        for hsmap in filter_maps.values():
            assert hsmap.dtype == np.int16

    def test_valid_pixels_have_positive_counts(self, filter_maps):
        for hsmap in filter_maps.values():
            valid = hsmap.valid_pixels
            assert len(valid) > 0
            assert np.all(hsmap[valid] >= 1)

    def test_all_filters_have_coverage(self, filter_maps):
        for filt, hsmap in filter_maps.items():
            assert len(hsmap.valid_pixels) > 0, f"Filter {filt} has no coverage"


class TestFilterAtPixels:
    def test_returns_valid_filter_names(self, filter_maps, cam_table):
        from argus_sim.coverage_map import filter_at_pixels

        az = np.array(cam_table["az_center"][:10])
        alt = np.array(cam_table["alt_center"][:10])
        result = filter_at_pixels(filter_maps, az, alt)
        assert len(result) == 10
        valid_filters = set(c.filter_strategy.options) | {""}
        for f in result:
            assert f in valid_filters

    def test_outside_footprint_returns_empty(self, filter_maps):
        from argus_sim.coverage_map import filter_at_pixels

        az = np.array([0.0, 1.0])
        alt = np.array([0.0, 0.5])
        result = filter_at_pixels(filter_maps, az, alt)
        assert np.all(result == "")

    def test_result_length_matches_input(self, filter_maps):
        from argus_sim.coverage_map import filter_at_pixels

        n = 50
        az = np.random.uniform(0, 360, n)
        alt = np.random.uniform(-90, 90, n)
        result = filter_at_pixels(filter_maps, az, alt)
        assert len(result) == n


class TestBuildFilterMapsOrder:
    def test_keys_sorted_and_process_independent(self, filter_maps):
        assert list(filter_maps.keys()) == sorted(filter_maps.keys())


class TestRowsPerFilter:
    @pytest.fixture(scope="class")
    def pixel_table(self, cam_table):
        import astropy.table as tbl

        rng = np.random.default_rng(11)
        n = 4000
        idx = rng.integers(0, len(cam_table), n)
        az = np.asarray(cam_table["az_center"])[idx] + rng.uniform(-1.5, 1.5, n)
        alt = np.clip(np.asarray(cam_table["alt_center"])[idx] + rng.uniform(-1.5, 1.5, n), 0, 90)
        return tbl.Table({"healpix": np.arange(n), "az": az % 360.0, "alt": alt})

    def test_one_row_per_pixel_and_covering_filter(self, filter_maps, pixel_table):
        from argus_sim.coverage_map import rows_per_filter

        out = rows_per_filter(pixel_table, filter_maps)
        az, alt = pixel_table["az"].data, pixel_table["alt"].data
        n_cover = sum((m.get_values_pos(az, alt, lonlat=True) > 0).astype(int) for m in filter_maps.values())
        assert len(out) == int(np.sum(n_cover))
        pairs = list(zip(out["healpix"], out["filter"]))
        assert len(pairs) == len(set(pairs))

    def test_contested_pixels_carry_every_covering_band(self, filter_maps, pixel_table):
        from argus_sim.coverage_map import rows_per_filter

        out = rows_per_filter(pixel_table, filter_maps)
        az, alt = pixel_table["az"].data, pixel_table["alt"].data
        for filt, hsmap in filter_maps.items():
            covered = set(pixel_table["healpix"][hsmap.get_values_pos(az, alt, lonlat=True) > 0])
            got = set(out["healpix"][out["filter"] == filt])
            assert got == covered

    def test_uncovered_pixels_are_dropped(self, filter_maps):
        import astropy.table as tbl

        from argus_sim.coverage_map import rows_per_filter

        t = tbl.Table({"healpix": [1, 2], "az": [0.0, 1.0], "alt": [0.0, 0.5]})
        out = rows_per_filter(t, filter_maps)
        assert len(out) == 0
        assert "filter" in out.colnames

    def test_band_row_share_tracks_camera_share(self, filter_maps, cam_table, pixel_table):
        """Row counts per band follow the camera counts, not a tie-break."""
        from argus_sim.coverage_map import rows_per_filter

        out = rows_per_filter(pixel_table, filter_maps)
        out = out[out["filter"] != ""]
        cams = np.asarray(cam_table["filter"])
        for filt in filter_maps:
            row_share = np.mean(out["filter"] == filt)
            cam_share = np.mean(cams == filt)
            assert row_share == pytest.approx(cam_share, abs=0.05), f"{filt}: rows {row_share:.3f} cams {cam_share:.3f}"


class TestBoxOrientation:
    """A rot = 0 camera is wider in azimuth than in altitude (long axis along the ring).

    The layout generator lays the 3.253 deg axis tangentially along each ring;
    mocpy's ``angle`` is measured from north to the semi-major axis, so the
    boxes must be built with angle = rot + 90 deg.
    """

    def test_single_camera_long_axis_along_azimuth(self):
        import astropy.table as tbl

        from argus_sim.coverage_map import build_filter_maps

        cam = tbl.Table({"alt_center": [60.0], "az_center": [180.0], "rot_deg": [0.0], "filter": ["g"]})
        maps = build_filter_maps(cam, 3.259 * u.deg, 2.446 * u.deg, 10, 1024)
        m = maps["g"]
        az, alt = hsp.HealSparseMap.valid_pixels_pos(m, lonlat=True)
        az_extent = (np.ptp(az) * np.cos(np.radians(60.0))) if len(az) else 0.0
        alt_extent = np.ptp(alt)
        assert az_extent == pytest.approx(3.259, abs=0.15)
        assert alt_extent == pytest.approx(2.446, abs=0.15)
        assert az_extent > alt_extent

    def test_ring_footprint_area_matches_design_orientation(self):
        """Instantaneous coverage of the canonical layout with long axes along the rings."""
        layout = load_ring_layout("A170r_1200")
        _, area, cam, _ = ring_pointing(layout, 3.259 * u.deg, 2.446 * u.deg, 10)
        # exact polygon clipping / point sampling give 8066 deg2;
        # the depth-10 MOC union converges to it from above
        assert 8040 < area.value < 8120, area
        assert "box_angle_deg" in cam.colnames


class TestRatchetGridSampling:
    """Per-ratchet coverage is the footprint, stable across ratchets.

    ``to_hpx_table_at`` samples the ratchet-time CIRS grid directly against the
    alt/az filter maps; moving the reference-epoch cell centres and re-binning
    them would alias 0-9% of the footprint away, depending on the grid phase.
    """

    def test_coverage_equals_footprint_and_is_stable(self):
        import astropy.time as atime

        from argus_sim.grid import CradlePointing

        old_depth = c.survey.sim_depth
        c.survey.sim_depth = 7
        try:
            cp = CradlePointing()
        finally:
            c.survey.sim_depth = old_depth
        union = len(np.unique(np.concatenate([cp.filter_maps[b].valid_pixels for b in cp.filter_maps])))
        counts = []
        for mjd in (61206.20, 61206.27, 61206.35, 61300.10):
            tab = cp.to_hpx_table_at(atime.Time(mjd, format="mjd"))
            counts.append(len(np.unique(np.asarray(tab["healpix"]))))
            assert set(np.asarray(tab["filter"]).astype(str)) == set(cp.filter_maps)
        counts = np.array(counts, dtype=float)
        assert np.ptp(counts) / counts.mean() < 0.005, counts
        assert abs(counts.mean() - union) / union < 0.005, (counts, union)
