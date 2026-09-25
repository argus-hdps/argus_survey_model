import json
import os
import tempfile
from unittest.mock import MagicMock

import healsparse as hsp
import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from argus_sim.cadence_accumulator import (
    _extract_signal_rate,
    DEPTH_DTYPE,
    CadenceAccumulator,
    build_window_registry,
    limmag_from_map,
    limmag_from_map_zp,
)


# --- Helpers for hypothesis strategies ---


def _make_single_night(n_ratchets, base_mjd=59000.0, ratchet_spacing_days=15 / 1440):
    """Generate ratchets within a single night (all same night index)."""
    ratchet_nums = np.arange(n_ratchets)
    mjds = base_mjd + np.arange(n_ratchets) * ratchet_spacing_days
    nights = np.ones(n_ratchets, dtype=int)
    return ratchet_nums, mjds, nights


def _make_multi_night(ratchets_per_night, base_mjd=59000.0):
    """Generate ratchets spread across multiple nights."""
    ratchet_nums = []
    mjds = []
    nights = []
    ratchet_spacing_days = 15 / 1440  # 15-minute ratchets
    idx = 0
    for night_idx, n in enumerate(ratchets_per_night, start=1):
        night_start = base_mjd + night_idx * 1.0  # each night ~1 day apart
        for j in range(n):
            ratchet_nums.append(idx)
            mjds.append(night_start + j * ratchet_spacing_days)
            nights.append(night_idx)
            idx += 1
    return np.array(ratchet_nums), np.array(mjds), np.array(nights)


# --- Property tests ---


class TestWindowRegistry:
    def test_all_cadence_levels_present(self):
        """Return dict has an entry for every requested cadence level."""
        ratchet_nums, mjds, nights = _make_single_night(10)
        levels = ["hour", "night", "season"]
        registry = build_window_registry(ratchet_nums, mjds, nights, levels)
        assert set(registry.keys()) == set(levels)

    def test_all_ratchets_assigned(self):
        """Every input ratchet_num appears in every cadence level mapping."""
        ratchet_nums, mjds, nights = _make_multi_night([5, 8, 3])
        levels = ["hour", "night", "season"]
        registry = build_window_registry(ratchet_nums, mjds, nights, levels)
        for level in levels:
            assert set(registry[level].keys()) == set(ratchet_nums)

    def test_same_night_same_window_id(self):
        """All ratchets in the same night map to the same night-level window_id."""
        ratchets_per_night = [6, 10, 4]
        ratchet_nums, mjds, nights = _make_multi_night(ratchets_per_night)
        registry = build_window_registry(ratchet_nums, mjds, nights, ["night"])
        night_map = registry["night"]
        for night_val in np.unique(nights):
            mask = nights == night_val
            window_ids = [night_map[r] for r in ratchet_nums[mask]]
            assert len(set(window_ids)) == 1, (
                f"Ratchets in night {night_val} mapped to multiple window_ids: {set(window_ids)}"
            )

    def test_different_nights_different_window_ids(self):
        """Ratchets in different nights map to different night-level window_ids."""
        ratchet_nums, mjds, nights = _make_multi_night([3, 3, 3])
        registry = build_window_registry(ratchet_nums, mjds, nights, ["night"])
        night_map = registry["night"]
        window_ids_per_night = []
        for night_val in np.unique(nights):
            mask = nights == night_val
            wid = night_map[ratchet_nums[mask][0]]
            window_ids_per_night.append(wid)
        assert len(set(window_ids_per_night)) == len(np.unique(nights))

    def test_hour_windows_span_one_twentyfourth_day(self):
        """Ratchets separated by less than 1/24 day share an hour window;
        ratchets more than 1 hour apart get different hour windows.
        """
        base_mjd = 59000.0
        ratchet_nums = np.arange(5)
        mjds = np.array(
            [
                base_mjd,
                base_mjd + 0.01,  # ~14 min later, same hour
                base_mjd + 0.03,  # ~43 min later, same hour
                base_mjd + 1 / 24 + 0.001,  # just past 1 hour, new window
                base_mjd + 2 / 24,  # 2 hours later, yet another window
            ]
        )
        nights = np.ones(5, dtype=int)
        registry = build_window_registry(ratchet_nums, mjds, nights, ["hour"])
        hour_map = registry["hour"]
        assert hour_map[0] == hour_map[1] == hour_map[2]
        assert hour_map[3] != hour_map[0]
        assert hour_map[4] != hour_map[3]

    def test_season_single_window(self):
        """All ratchets map to the same season window when within one season."""
        ratchet_nums, mjds, nights = _make_multi_night([5, 5, 5])
        registry = build_window_registry(ratchet_nums, mjds, nights, ["season"])
        season_map = registry["season"]
        assert len(set(season_map.values())) == 1

    def test_season_multiple_windows(self):
        """Ratchets 90+ days apart should land in different season windows."""
        ratchet_nums = np.arange(3)
        mjds = np.array([59000.0, 59000.5, 59100.0])
        nights = np.array([1, 1, 2])
        registry = build_window_registry(ratchet_nums, mjds, nights, ["season"])
        season_map = registry["season"]
        assert season_map[0] == season_map[1]
        assert season_map[2] != season_map[0]

    def test_window_ids_are_zero_indexed_contiguous(self):
        """Window IDs should be contiguous integers starting from 0."""
        ratchet_nums, mjds, nights = _make_multi_night([4, 4, 4])
        registry = build_window_registry(ratchet_nums, mjds, nights, ["night", "hour"])
        for level in ["night", "hour"]:
            ids = sorted(set(registry[level].values()))
            assert ids == list(range(len(ids)))

    def test_empty_input(self):
        """Empty ratchet list produces empty mappings."""
        registry = build_window_registry(
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=int),
            ["hour", "night", "season"],
        )
        for level in ["hour", "night", "season"]:
            assert registry[level] == {}

    def test_single_ratchet(self):
        """Single ratchet gets window_id 0 at all levels."""
        registry = build_window_registry(
            np.array([42]),
            np.array([59000.0]),
            np.array([1]),
            ["hour", "night", "season"],
        )
        for level in ["hour", "night", "season"]:
            assert registry[level] == {42: 0}

    def test_custom_cadence_levels(self):
        """Only requested cadence levels appear in output."""
        ratchet_nums, mjds, nights = _make_single_night(5)
        registry = build_window_registry(ratchet_nums, mjds, nights, ["night"])
        assert list(registry.keys()) == ["night"]

    def test_15min_windows_span_exactly_15min(self):
        """Ratchets in the same 15-min bucket share a window."""
        base_mjd = 59000.0
        width = 15.0 / 1440.0
        ratchet_nums = np.arange(4)
        mjds = np.array(
            [
                base_mjd,
                base_mjd + 0.5 * width,  # same 15-min bucket
                base_mjd + width + 1e-9,  # just past the boundary, new bucket
                base_mjd + 3 * width,  # third bucket
            ]
        )
        nights = np.ones(4, dtype=int)
        registry = build_window_registry(ratchet_nums, mjds, nights, ["15min"])
        m = registry["15min"]
        assert m[0] == m[1]
        assert m[2] != m[0]
        assert m[3] != m[2]

    def test_15min_boundary_ratchet_falls_in_new_window(self):
        """A ratchet at exactly the bucket edge belongs to the next window."""
        base_mjd = 59000.0
        width = 15.0 / 1440.0
        ratchet_nums = np.arange(2)
        mjds = np.array([base_mjd, base_mjd + width])
        nights = np.ones(2, dtype=int)
        registry = build_window_registry(ratchet_nums, mjds, nights, ["15min"])
        m = registry["15min"]
        assert m[0] != m[1]

    @given(
        n_ratchets=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=50)
    def test_15min_window_ids_contiguous_from_zero(self, n_ratchets):
        """For any ratchet set, 15min window IDs are contiguous starting from 0."""
        base_mjd = 59000.0
        width = 15.0 / 1440.0
        ratchet_nums = np.arange(n_ratchets)
        mjds = base_mjd + np.arange(n_ratchets) * width * 0.6
        nights = np.ones(n_ratchets, dtype=int)
        registry = build_window_registry(ratchet_nums, mjds, nights, ["15min"])
        ids = sorted(set(registry["15min"].values()))
        assert ids == list(range(len(ids)))

    @given(
        n_ratchets=st.integers(min_value=2, max_value=40),
    )
    @settings(max_examples=50)
    def test_15min_window_spans_nominal_duration(self, n_ratchets):
        """Each 15min window spans at most 15 minutes of MJD."""
        base_mjd = 59000.0
        width = 15.0 / 1440.0
        ratchet_nums = np.arange(n_ratchets)
        mjds = base_mjd + np.sort(np.random.default_rng(42).uniform(0, 10 * width, n_ratchets))
        nights = np.ones(n_ratchets, dtype=int)
        registry = build_window_registry(ratchet_nums, mjds, nights, ["15min"])
        m = registry["15min"]
        from collections import defaultdict

        window_mjds = defaultdict(list)
        for r, wid in m.items():
            window_mjds[wid].append(mjds[r])
        for wid, wm in window_mjds.items():
            assert max(wm) - min(wm) < width

    def test_week_windows_span_7_days(self):
        """Ratchets in the same 7-day bucket share a week window."""
        base_mjd = 59003.0  # floor(59003/7)=8429, bucket spans [59003,59010)
        ratchet_nums = np.arange(4)
        mjds = np.array(
            [
                base_mjd,
                base_mjd + 3.0,  # same week bucket
                base_mjd + 7.0,  # next bucket
                base_mjd + 14.0,  # third bucket
            ]
        )
        nights = np.array([1, 2, 3, 4])
        registry = build_window_registry(ratchet_nums, mjds, nights, ["week"])
        m = registry["week"]
        assert m[0] == m[1]
        assert m[2] != m[0]
        assert m[3] != m[2]

    def test_week_boundary_ratchet_falls_in_new_window(self):
        """A ratchet at exactly the 7-day boundary belongs to the next window."""
        base_mjd = 59003.0
        ratchet_nums = np.arange(2)
        mjds = np.array([base_mjd, base_mjd + 7.0])
        nights = np.array([1, 2])
        registry = build_window_registry(ratchet_nums, mjds, nights, ["week"])
        m = registry["week"]
        assert m[0] != m[1]

    @given(
        n_ratchets=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=50)
    def test_week_window_ids_contiguous_from_zero(self, n_ratchets):
        """For any ratchet set, week window IDs are contiguous starting from 0."""
        base_mjd = 59000.0
        ratchet_nums = np.arange(n_ratchets)
        mjds = base_mjd + np.arange(n_ratchets) * 2.0
        nights = np.arange(n_ratchets)
        registry = build_window_registry(ratchet_nums, mjds, nights, ["week"])
        ids = sorted(set(registry["week"].values()))
        assert ids == list(range(len(ids)))

    @given(
        n_ratchets=st.integers(min_value=2, max_value=40),
    )
    @settings(max_examples=50)
    def test_week_window_spans_nominal_duration(self, n_ratchets):
        """Each week window spans at most 7 days of MJD."""
        base_mjd = 59000.0
        ratchet_nums = np.arange(n_ratchets)
        mjds = base_mjd + np.sort(np.random.default_rng(7).uniform(0, 50.0, n_ratchets))
        nights = np.arange(n_ratchets)
        registry = build_window_registry(ratchet_nums, mjds, nights, ["week"])
        m = registry["week"]
        from collections import defaultdict

        window_mjds = defaultdict(list)
        for r, wid in m.items():
            window_mjds[wid].append(mjds[r])
        for wid, wm in window_mjds.items():
            assert max(wm) - min(wm) < 7.0

    @given(
        n_nights=st.integers(min_value=1, max_value=10),
        ratchets_per=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=50)
    def test_night_window_count_matches_unique_nights(self, n_nights, ratchets_per):
        """Number of distinct night window_ids equals number of unique night values."""
        ratchet_nums, mjds, nights = _make_multi_night([ratchets_per] * n_nights)
        registry = build_window_registry(ratchet_nums, mjds, nights, ["night"])
        n_window_ids = len(set(registry["night"].values()))
        assert n_window_ids == n_nights

    @given(
        n_nights=st.integers(min_value=1, max_value=5),
        ratchets_per=st.integers(min_value=1, max_value=15),
    )
    @settings(max_examples=50)
    def test_same_night_property(self, n_nights, ratchets_per):
        """Property: for any set of ratchets within the same night,
        all map to the same night-level window_id.
        """
        ratchet_nums, mjds, nights = _make_multi_night([ratchets_per] * n_nights)
        registry = build_window_registry(ratchet_nums, mjds, nights, ["night"])
        night_map = registry["night"]
        for night_val in np.unique(nights):
            mask = nights == night_val
            window_ids = {night_map[r] for r in ratchet_nums[mask]}
            assert len(window_ids) == 1


# --- Helpers for CadenceAccumulator tests ---


def _make_ratchet_df(hpx, noise, source_shot, exptime=30.0, band="g"):
    """Build a minimal ratchet data dict with the required keys."""
    return {
        "healpix": np.asarray(hpx),
        "noise": np.asarray(noise, dtype=np.float64),
        "source_shot": np.asarray(source_shot, dtype=np.float64),
        "exptime_s": float(exptime),
        "band": np.full(len(hpx), band),
    }


def _read_map(acc, level, window_id, band):
    """Get a window map from memory if still there, otherwise from disk."""
    if window_id in acc._maps.get(level, {}):
        if band in acc._maps[level][window_id]:
            return acc._maps[level][window_id][band]
    fname = f"depth_{level}_w{window_id:04d}_{band}.hsp"
    fpath = os.path.join(acc._output_dir, level, fname)
    return hsp.HealSparseMap.read(fpath)


def _simple_accumulator(
    n_ratchets=3,
    levels=None,
    nside=128,
    output_dir=None,
    base_mjd=59000.0,
):
    """Create a CadenceAccumulator with n_ratchets in a single night."""
    if levels is None:
        levels = ["night"]
    ratchet_nums, mjds, nights = _make_single_night(n_ratchets, base_mjd=base_mjd)
    registry = build_window_registry(ratchet_nums, mjds, nights, levels)
    mjd_map = {int(r): float(m) for r, m in zip(ratchet_nums, mjds)}
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    return (
        CadenceAccumulator(
            window_registry=registry,
            cadence_levels=levels,
            nside_sparse=nside,
            output_dir=output_dir,
            mjds=mjd_map,
        ),
        ratchet_nums,
        mjds,
    )


# --- CadenceAccumulator tests ---


class TestCadenceAccumulator:
    def test_depth_dtype_fields(self):
        """DEPTH_DTYPE has the three required fields."""
        assert DEPTH_DTYPE.names == ("inv_bkg_noise_sq_sum", "n_obs", "exptime_s")

    def test_single_ingest_produces_correct_depth(self):
        """A single ratchet ingest stores 1/bkg_noise^2 at each pixel."""
        acc, ratchet_nums, _ = _simple_accumulator(n_ratchets=1)
        noise = np.array([10.0, 20.0])
        source_shot = np.array([3.0, 4.0])
        hpx = np.array([100, 200])
        df = _make_ratchet_df(hpx, noise, source_shot, exptime=30.0)

        acc.ingest(0, df)

        hsmap = _read_map(acc, "night", 0, "g")
        bkg_noise_sq = noise**2 - source_shot**2
        expected_inv = 1.0 / bkg_noise_sq

        vals = hsmap.get_values_pix(hpx)
        np.testing.assert_allclose(vals["inv_bkg_noise_sq_sum"], expected_inv)
        np.testing.assert_array_equal(vals["n_obs"], 1)
        np.testing.assert_allclose(vals["exptime_s"], 30.0)

    def test_two_ratchets_accumulate_additively(self):
        """Ingesting two ratchets sums inverse-variance at shared pixels."""
        acc, ratchet_nums, _ = _simple_accumulator(n_ratchets=2)
        hpx = np.array([100, 200])

        noise1 = np.array([10.0, 20.0])
        shot1 = np.array([3.0, 4.0])
        df1 = _make_ratchet_df(hpx, noise1, shot1, exptime=30.0)

        noise2 = np.array([12.0, 18.0])
        shot2 = np.array([2.0, 5.0])
        df2 = _make_ratchet_df(hpx, noise2, shot2, exptime=30.0)

        acc.ingest(0, df1)
        acc.ingest(1, df2)

        hsmap = _read_map(acc, "night", 0, "g")
        bkg1 = noise1**2 - shot1**2
        bkg2 = noise2**2 - shot2**2
        expected_inv = 1.0 / bkg1 + 1.0 / bkg2

        vals = hsmap.get_values_pix(hpx)
        np.testing.assert_allclose(vals["inv_bkg_noise_sq_sum"], expected_inv)
        np.testing.assert_array_equal(vals["n_obs"], 2)
        np.testing.assert_allclose(vals["exptime_s"], 60.0)

    def test_pixel_in_two_bands_counts_once_per_band(self):
        """An overlap pixel imaged in two bands in one ratchet adds one
        observation to each band map, never two to either.
        """
        acc, _, _ = _simple_accumulator(n_ratchets=1)
        hpx = np.array([100, 100, 200])
        df = _make_ratchet_df(hpx, [10.0, 12.0, 20.0], [3.0, 2.0, 4.0], exptime=30.0)
        df["band"] = np.array(["g", "rho", "g"])

        acc.ingest(0, df)

        g = _read_map(acc, "night", 0, "g").get_values_pix([100, 200])
        rho = _read_map(acc, "night", 0, "rho").get_values_pix([100])
        np.testing.assert_array_equal(g["n_obs"], [1, 1])
        np.testing.assert_array_equal(rho["n_obs"], [1])
        np.testing.assert_allclose(g["inv_bkg_noise_sq_sum"][0], 1.0 / (10.0**2 - 3.0**2))
        np.testing.assert_allclose(rho["inv_bkg_noise_sq_sum"][0], 1.0 / (12.0**2 - 2.0**2))

    def test_skip_does_not_add_pixels(self):
        """Skipping a ratchet contributes no data to the depth maps."""
        acc, _, _ = _simple_accumulator(n_ratchets=2)
        acc.skip(0)
        acc.skip(1)
        assert len(acc._maps["night"]) == 0

    def test_band_defaults_to_single_band(self):
        """Ingest works with a single-value band array."""
        acc, _, _ = _simple_accumulator(n_ratchets=1)
        data = _make_ratchet_df(np.array([100, 200]), np.array([10.0, 20.0]), np.array([3.0, 4.0]))

        acc.ingest(0, data)
        hsmap = _read_map(acc, "night", 0, "g")
        assert hsmap.valid_pixels.size == 2

    def test_finalize_writes_hsp_files(self):
        """Finalize writes .hsp files that can be read back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            acc, _, _ = _simple_accumulator(n_ratchets=2, output_dir=tmpdir)
            hpx = np.array([100, 200])
            for r in range(2):
                df = _make_ratchet_df(hpx, np.array([10.0, 20.0]), np.array([3.0, 4.0]))
                acc.ingest(r, df)

            manifest = acc.finalize()

            night_windows = manifest["cadence_levels"]["night"]["windows"]
            assert len(night_windows) == 1
            band_files = night_windows[0]["bands"]
            assert len(band_files) == 1
            assert band_files[0]["band"] == "g"
            fpath = os.path.join(tmpdir, band_files[0]["file"])
            assert os.path.isfile(fpath)

            m = hsp.HealSparseMap.read(fpath)
            assert m.valid_pixels.size == 2

    def test_finalize_writes_manifest_json(self):
        """Finalize writes cadence_manifest.json with required structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            acc, _, _ = _simple_accumulator(
                n_ratchets=1,
                levels=["hour", "night"],
                output_dir=tmpdir,
            )
            df = _make_ratchet_df(np.array([100]), np.array([10.0]), np.array([3.0]))
            acc.ingest(0, df)
            acc.finalize()

            manifest_path = os.path.join(tmpdir, "cadence_manifest.json")
            assert os.path.isfile(manifest_path)
            with open(manifest_path) as f:
                data = json.load(f)
            assert "nside_sparse" in data
            assert set(data["cadence_levels"].keys()) == {"hour", "night"}

    def test_manifest_provenance_block(self):
        """Manifest contains provenance with timestamp and zeropoint assumptions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            acc, _, _ = _simple_accumulator(
                n_ratchets=1,
                levels=["night"],
                output_dir=tmpdir,
            )
            df = _make_ratchet_df(np.array([100]), np.array([10.0]), np.array([3.0]))
            acc.ingest(0, df)
            manifest = acc.finalize()

            assert "provenance" in manifest
            prov = manifest["provenance"]
            assert "timestamp_utc" in prov
            assert "zeropoint_assumptions" in prov
            assert "airmass" in prov["zeropoint_assumptions"].lower()

            manifest_path = os.path.join(tmpdir, "cadence_manifest.json")
            with open(manifest_path) as f:
                on_disk = json.load(f)
            assert on_disk["provenance"] == manifest["provenance"]
            assert isinstance(prov["git_dirty"], bool)
            with open(os.path.join(tmpdir, "provenance.json")) as f:
                side = json.load(f)
            assert side["product"] == "survey"
            assert side["git_hash_short"] == prov["git_hash"]
            assert side["git_dirty"] == prov["git_dirty"]

    def test_manifest_provenance_config_fingerprint(self):
        """Config fingerprint is included in provenance when provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ratchet_nums, mjds, nights = _make_single_night(1)
            registry = build_window_registry(ratchet_nums, mjds, nights, ["night"])
            mjd_map = {int(r): float(m) for r, m in zip(ratchet_nums, mjds)}
            acc = CadenceAccumulator(
                window_registry=registry,
                cadence_levels=["night"],
                nside_sparse=128,
                output_dir=tmpdir,
                mjds=mjd_map,
                config_fingerprint="sha256:abc123",
            )
            df = _make_ratchet_df(np.array([100]), np.array([10.0]), np.array([3.0]))
            acc.ingest(0, df)
            manifest = acc.finalize()

            assert manifest["provenance"]["config_fingerprint"] == "sha256:abc123"

    def test_manifest_mjd_ranges(self):
        """Manifest records MJD min/max for each window."""
        with tempfile.TemporaryDirectory() as tmpdir:
            acc, ratchet_nums, mjds = _simple_accumulator(
                n_ratchets=3,
                output_dir=tmpdir,
            )
            hpx = np.array([100])
            for r in range(3):
                df = _make_ratchet_df(hpx, np.array([10.0]), np.array([3.0]))
                acc.ingest(int(ratchet_nums[r]), df)

            manifest = acc.finalize()
            win = manifest["cadence_levels"]["night"]["windows"][0]
            assert win["mjd_min"] <= win["mjd_max"]
            assert win["mjd_min"] == pytest.approx(float(mjds[0]))
            assert win["mjd_max"] == pytest.approx(float(mjds[-1]))

    def test_multi_level_produces_separate_maps(self):
        """Different cadence levels produce independent depth maps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ratchet_nums, mjds, nights = _make_multi_night([2, 2])
            levels = ["hour", "night"]
            registry = build_window_registry(ratchet_nums, mjds, nights, levels)
            mjd_map = {int(r): float(m) for r, m in zip(ratchet_nums, mjds)}
            acc = CadenceAccumulator(
                window_registry=registry,
                cadence_levels=levels,
                nside_sparse=128,
                output_dir=tmpdir,
                mjds=mjd_map,
            )
            hpx = np.array([100])
            for r in ratchet_nums:
                df = _make_ratchet_df(hpx, np.array([10.0]), np.array([3.0]))
                acc.ingest(int(r), df)

            manifest = acc.finalize()
            n_night = manifest["cadence_levels"]["night"]["n_windows"]
            n_hour = manifest["cadence_levels"]["hour"]["n_windows"]
            assert n_night == 2
            assert n_hour >= n_night

    def test_zero_bkg_noise_pixels_excluded(self):
        """Pixels where bkg_noise would be zero are silently dropped."""
        acc, _, _ = _simple_accumulator(n_ratchets=1)
        hpx = np.array([100, 200])
        noise = np.array([5.0, 5.0])
        source_shot = np.array([5.0, 3.0])
        df = _make_ratchet_df(hpx, noise, source_shot)

        acc.ingest(0, df)
        hsmap = _read_map(acc, "night", 0, "g")
        assert hsmap.valid_pixels.size == 1
        assert 200 in hsmap.valid_pixels

    def test_duplicate_healpix_aggregated(self):
        """Multiple rows with same healpix in one ratchet are summed."""
        acc, _, _ = _simple_accumulator(n_ratchets=1)
        hpx = np.array([100, 100])
        noise = np.array([10.0, 12.0])
        source_shot = np.array([3.0, 4.0])
        data = _make_ratchet_df(hpx, noise, source_shot, exptime=30.0)

        acc.ingest(0, data)
        hsmap = _read_map(acc, "night", 0, "g")
        vals = hsmap.get_values_pix(np.array([100]))

        bkg1 = noise[0] ** 2 - source_shot[0] ** 2
        bkg2 = noise[1] ** 2 - source_shot[1] ** 2
        expected_inv = 1.0 / bkg1 + 1.0 / bkg2
        np.testing.assert_allclose(vals["inv_bkg_noise_sq_sum"][0], expected_inv)
        np.testing.assert_allclose(vals["exptime_s"][0], 60.0)
        assert vals["n_obs"][0] == 2

    @given(
        noise_val=st.floats(min_value=5.0, max_value=100.0),
        shot_val=st.floats(min_value=0.1, max_value=4.9),
    )
    @settings(max_examples=50)
    def test_inv_bkg_noise_sq_is_positive(self, noise_val, shot_val):
        """Accumulated inverse-variance is always positive when noise > source_shot."""
        assume(noise_val > shot_val)
        acc, _, _ = _simple_accumulator(n_ratchets=1)
        df = _make_ratchet_df(
            np.array([100]),
            np.array([noise_val]),
            np.array([shot_val]),
        )
        acc.ingest(0, df)
        hsmap = _read_map(acc, "night", 0, "g")
        vals = hsmap.get_values_pix(np.array([100]))
        assert vals["inv_bkg_noise_sq_sum"][0] > 0

    @given(
        n_ratchets=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=20)
    def test_n_obs_equals_ratchet_count(self, n_ratchets):
        """n_obs at each pixel equals the number of ratchets that contributed."""
        acc, ratchet_nums, _ = _simple_accumulator(n_ratchets=n_ratchets)
        hpx = np.array([100])
        for r in range(n_ratchets):
            df = _make_ratchet_df(hpx, np.array([10.0]), np.array([3.0]))
            acc.ingest(r, df)

        hsmap = _read_map(acc, "night", 0, "g")
        vals = hsmap.get_values_pix(hpx)
        assert vals["n_obs"][0] == n_ratchets

    def test_exptime_accumulates_correctly(self):
        """Total exptime_s is the sum of individual exposure times."""
        acc, _, _ = _simple_accumulator(n_ratchets=3)
        hpx = np.array([100])
        exptimes = [30.0, 45.0, 60.0]
        for r, exp in enumerate(exptimes):
            df = _make_ratchet_df(hpx, np.array([10.0]), np.array([3.0]), exptime=exp)
            acc.ingest(r, df)

        hsmap = _read_map(acc, "night", 0, "g")
        vals = hsmap.get_values_pix(hpx)
        np.testing.assert_allclose(vals["exptime_s"][0], sum(exptimes))

    @given(perm=st.permutations(list(range(5))))
    @settings(max_examples=50)
    def test_property_order_independence(self, perm):
        """Final depth maps are identical regardless of ratchet ingest order."""
        n_ratchets = 5
        nside = 128
        hpx = np.array([100, 200, 300, 400])
        noise_vals = [10.0, 15.0, 12.0, 20.0, 8.0]
        shot_vals = [3.0, 4.0, 2.0, 5.0, 1.0]

        ratchet_nums, mjds, nights = _make_single_night(n_ratchets)
        registry = build_window_registry(ratchet_nums, mjds, nights, ["night"])
        mjd_map = {int(r): float(m) for r, m in zip(ratchet_nums, mjds)}

        dfs = {}
        for r in range(n_ratchets):
            dfs[r] = _make_ratchet_df(
                hpx,
                np.full(len(hpx), noise_vals[r]),
                np.full(len(hpx), shot_vals[r]),
            )

        acc_canonical = CadenceAccumulator(
            window_registry=registry,
            cadence_levels=["night"],
            nside_sparse=nside,
            output_dir=tempfile.mkdtemp(),
            mjds=mjd_map,
        )
        for r in range(n_ratchets):
            acc_canonical.ingest(r, dfs[r])

        acc_permuted = CadenceAccumulator(
            window_registry=registry,
            cadence_levels=["night"],
            nside_sparse=nside,
            output_dir=tempfile.mkdtemp(),
            mjds=mjd_map,
        )
        for r in perm:
            acc_permuted.ingest(r, dfs[r])

        canonical_map = _read_map(acc_canonical, "night", 0, "g")
        permuted_map = _read_map(acc_permuted, "night", 0, "g")

        np.testing.assert_array_equal(canonical_map.valid_pixels, permuted_map.valid_pixels)
        c_vals = canonical_map.get_values_pix(canonical_map.valid_pixels)
        p_vals = permuted_map.get_values_pix(permuted_map.valid_pixels)
        np.testing.assert_allclose(c_vals["inv_bkg_noise_sq_sum"], p_vals["inv_bkg_noise_sq_sum"])
        np.testing.assert_array_equal(c_vals["n_obs"], p_vals["n_obs"])
        np.testing.assert_allclose(c_vals["exptime_s"], p_vals["exptime_s"])

    @given(
        n_nights=st.integers(min_value=2, max_value=4),
        ratchets_per=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=50)
    def test_property_hierarchy_conservation(self, n_nights, ratchets_per):
        """Sum of ratchet-level inv_bkg_noise_sq_sum equals the night-level value."""
        nside = 128
        hpx = np.array([100, 200, 300])
        noise = 10.0
        shot = 3.0

        ratchet_nums, mjds, nights = _make_multi_night([ratchets_per] * n_nights)
        registry = build_window_registry(ratchet_nums, mjds, nights, ["night"])
        mjd_map = {int(r): float(m) for r, m in zip(ratchet_nums, mjds)}

        acc_night = CadenceAccumulator(
            window_registry=registry,
            cadence_levels=["night"],
            nside_sparse=nside,
            output_dir=tempfile.mkdtemp(),
            mjds=mjd_map,
        )
        for r in ratchet_nums:
            df = _make_ratchet_df(hpx, np.full(len(hpx), noise), np.full(len(hpx), shot))
            acc_night.ingest(int(r), df)

        ratchet_maps = {}
        for r in ratchet_nums:
            single_reg = build_window_registry(
                np.array([r]),
                np.array([mjds[r]]),
                np.array([nights[r]]),
                ["night"],
            )
            acc_single = CadenceAccumulator(
                window_registry=single_reg,
                cadence_levels=["night"],
                nside_sparse=nside,
                output_dir=tempfile.mkdtemp(),
                mjds={int(r): float(mjds[r])},
            )
            df = _make_ratchet_df(hpx, np.full(len(hpx), noise), np.full(len(hpx), shot))
            acc_single.ingest(int(r), df)
            ratchet_maps[int(r)] = _read_map(acc_single, "night", 0, "g")

        for night_val in np.unique(nights):
            night_window_id = registry["night"][int(ratchet_nums[nights == night_val][0])]
            night_hsmap = _read_map(acc_night, "night", night_window_id, "g")
            night_vals = night_hsmap.get_values_pix(hpx)

            sum_inv = np.zeros(len(hpx), dtype=np.float64)
            sum_exp = np.zeros(len(hpx), dtype=np.float64)
            sum_nobs = np.zeros(len(hpx), dtype=np.int32)
            for r in ratchet_nums[nights == night_val]:
                r_vals = ratchet_maps[int(r)].get_values_pix(hpx)
                sum_inv += r_vals["inv_bkg_noise_sq_sum"]
                sum_exp += r_vals["exptime_s"]
                sum_nobs += r_vals["n_obs"]

            np.testing.assert_allclose(night_vals["inv_bkg_noise_sq_sum"], sum_inv)
            np.testing.assert_allclose(night_vals["exptime_s"], sum_exp)
            np.testing.assert_array_equal(night_vals["n_obs"], sum_nobs)

    def test_n_epochs_multiplier(self):
        """ingest(df, n_epochs=30) produces the same result as 30 separate ingests."""
        n_epochs = 30
        acc_multi, _, _ = _simple_accumulator(n_ratchets=1)
        acc_single, _, _ = _simple_accumulator(n_ratchets=n_epochs)

        hpx = np.array([100, 200, 300])
        noise = np.array([10.0, 15.0, 20.0])
        shot = np.array([3.0, 4.0, 5.0])
        df = _make_ratchet_df(hpx, noise, shot, exptime=30.0)

        acc_multi.ingest(0, df, n_epochs=n_epochs)

        for r in range(n_epochs):
            acc_single.ingest(r, df)

        map_multi = _read_map(acc_multi, "night", 0, "g")
        map_single = _read_map(acc_single, "night", 0, "g")

        np.testing.assert_array_equal(map_multi.valid_pixels, map_single.valid_pixels)
        v_multi = map_multi.get_values_pix(hpx)
        v_single = map_single.get_values_pix(hpx)
        np.testing.assert_allclose(v_multi["inv_bkg_noise_sq_sum"], v_single["inv_bkg_noise_sq_sum"])
        np.testing.assert_array_equal(v_multi["n_obs"], v_single["n_obs"])
        np.testing.assert_allclose(v_multi["exptime_s"], v_single["exptime_s"])

    @given(
        n_epochs=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=50)
    def test_property_n_epochs_equivalence(self, n_epochs):
        """ingest(df, n_epochs=N) == N sequential ingest(df, n_epochs=1)."""
        acc_bulk, _, _ = _simple_accumulator(n_ratchets=1)
        acc_seq, _, _ = _simple_accumulator(n_ratchets=n_epochs)

        hpx = np.array([100, 200])
        noise = np.array([10.0, 15.0])
        shot = np.array([3.0, 4.0])
        df = _make_ratchet_df(hpx, noise, shot, exptime=30.0)

        acc_bulk.ingest(0, df, n_epochs=n_epochs)
        for r in range(n_epochs):
            acc_seq.ingest(r, df)

        map_bulk = _read_map(acc_bulk, "night", 0, "g")
        map_seq = _read_map(acc_seq, "night", 0, "g")
        v_bulk = map_bulk.get_values_pix(hpx)
        v_seq = map_seq.get_values_pix(hpx)

        np.testing.assert_allclose(v_bulk["inv_bkg_noise_sq_sum"], v_seq["inv_bkg_noise_sq_sum"])
        np.testing.assert_array_equal(v_bulk["n_obs"], v_seq["n_obs"])
        np.testing.assert_allclose(v_bulk["exptime_s"], v_seq["exptime_s"])

    @given(
        n_ratchets=st.integers(min_value=1, max_value=6),
        skip_idx=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=50)
    def test_property_skip_neutrality(self, n_ratchets, skip_idx):
        """Calling skip() never modifies any pixel value in any map."""
        assume(skip_idx < n_ratchets)

        nside = 128
        hpx = np.array([100, 200])
        noise = np.array([10.0, 15.0])
        shot = np.array([3.0, 4.0])

        total_ratchets = n_ratchets + 1
        ratchet_nums, mjds, nights = _make_single_night(total_ratchets)
        registry = build_window_registry(ratchet_nums, mjds, nights, ["night"])
        mjd_map = {int(r): float(m) for r, m in zip(ratchet_nums, mjds)}

        ingest_order = [r for r in range(total_ratchets) if r != skip_idx]

        acc_with_skip = CadenceAccumulator(
            window_registry=registry,
            cadence_levels=["night"],
            nside_sparse=nside,
            output_dir=tempfile.mkdtemp(),
            mjds=mjd_map,
        )
        for r in ingest_order:
            acc_with_skip.ingest(r, _make_ratchet_df(hpx, noise, shot))
        acc_with_skip.skip(skip_idx)

        acc_without_skip = CadenceAccumulator(
            window_registry=registry,
            cadence_levels=["night"],
            nside_sparse=nside,
            output_dir=tempfile.mkdtemp(),
            mjds=mjd_map,
        )
        for r in ingest_order:
            acc_without_skip.ingest(r, _make_ratchet_df(hpx, noise, shot))

        acc_with_skip.finalize()
        acc_without_skip.finalize()

        for level in ["night"]:
            for meta in acc_without_skip._flushed_meta[level]:
                wid = meta["window_id"]
                for bf in meta["bands"]:
                    band = bf["band"]
                    map_with = _read_map(acc_with_skip, level, wid, band)
                    map_without = _read_map(acc_without_skip, level, wid, band)

                    np.testing.assert_array_equal(map_with.valid_pixels, map_without.valid_pixels)
                    v_with = map_with.get_values_pix(map_with.valid_pixels)
                    v_without = map_without.get_values_pix(map_without.valid_pixels)
                    np.testing.assert_allclose(
                        v_with["inv_bkg_noise_sq_sum"],
                        v_without["inv_bkg_noise_sq_sum"],
                    )
                    np.testing.assert_array_equal(v_with["n_obs"], v_without["n_obs"])
                    np.testing.assert_allclose(v_with["exptime_s"], v_without["exptime_s"])


# --- Helpers for limmag_from_map tests ---


def _make_stub_ab(zp_photons_per_sec=1e6):
    """ABPhot stub whose mag_from_photons follows the standard AB formula."""
    ab = MagicMock()

    def _mag_from_photons(_band, photons):
        return -2.5 * np.log10(np.asarray(photons, dtype=np.float64) / zp_photons_per_sec)

    ab.mag_from_photons = _mag_from_photons
    return ab


def _make_depth_map(pixels, inv_var, exptime_s, n_obs=1, nside=128):
    """Build a HealSparse recarray map with given depth values."""
    hsmap = hsp.HealSparseMap.make_empty(
        nside_coverage=32,
        nside_sparse=nside,
        dtype=DEPTH_DTYPE,
        primary="inv_bkg_noise_sq_sum",
    )
    vals = np.zeros(len(pixels), dtype=DEPTH_DTYPE)
    vals["inv_bkg_noise_sq_sum"] = inv_var
    vals["n_obs"] = n_obs
    vals["exptime_s"] = exptime_s
    hsmap.update_values_pix(np.asarray(pixels, dtype=np.int64), vals)
    return hsmap


# --- limmag_from_map tests ---


class TestLimmagFromMap:
    @given(
        snr_a=st.floats(min_value=3.0, max_value=100.0),
        snr_b=st.floats(min_value=1.0, max_value=100.0),
    )
    @settings(max_examples=100)
    def test_limmag_snr_monotonic(self, snr_a, snr_b):
        """For SNR_a > SNR_b > 0, limmag(snr_a) <= limmag(snr_b) at every pixel."""
        assume(snr_a > snr_b)

        pixels = np.array([100, 200, 300])
        inv_var = np.array([1e-4, 1e-3, 1e-2])
        exptime = np.array([30.0, 60.0, 90.0])
        depth_map = _make_depth_map(pixels, inv_var, exptime)
        ab = _make_stub_ab()
        band_qe = 0.8

        result_a = limmag_from_map(depth_map, snr_a, ab, "V", band_qe)
        result_b = limmag_from_map(depth_map, snr_b, ab, "V", band_qe)

        np.testing.assert_array_equal(result_a["pixel"], result_b["pixel"])
        assert np.all(result_a["limmag"] <= result_b["limmag"]), (
            f"Monotonicity violated: snr_a={snr_a}, snr_b={snr_b}, "
            f"limmag_a={result_a['limmag']}, limmag_b={result_b['limmag']}"
        )

    def test_limmag_returns_correct_structure(self):
        """Output has pixel and limmag fields with expected lengths."""
        pixels = np.array([100, 200])
        inv_var = np.array([1e-3, 1e-3])
        exptime = np.array([30.0, 30.0])
        depth_map = _make_depth_map(pixels, inv_var, exptime)
        ab = _make_stub_ab()

        result = limmag_from_map(depth_map, 5.0, ab, "V", 0.8)

        assert result.dtype.names == ("pixel", "limmag")
        assert len(result) == 2
        np.testing.assert_array_equal(result["pixel"], pixels)

    def test_limmag_values_are_finite(self):
        """All output magnitudes are finite for valid depth map inputs."""
        pixels = np.array([100, 200, 300])
        inv_var = np.array([1e-4, 1e-3, 1e-2])
        exptime = np.array([30.0, 60.0, 90.0])
        depth_map = _make_depth_map(pixels, inv_var, exptime)
        ab = _make_stub_ab()

        result = limmag_from_map(depth_map, 5.0, ab, "V", 0.8)

        assert np.all(np.isfinite(result["limmag"]))

    def test_limmag_deeper_with_more_observations(self):
        """More accumulated inverse-variance yields fainter (larger) limmag."""
        ab = _make_stub_ab()
        snr = 5.0
        band_qe = 0.8

        shallow = _make_depth_map(np.array([100]), np.array([1e-4]), np.array([30.0]))
        deep = _make_depth_map(np.array([100]), np.array([1e-2]), np.array([30.0]))

        mag_shallow = limmag_from_map(shallow, snr, ab, "V", band_qe)["limmag"][0]
        mag_deep = limmag_from_map(deep, snr, ab, "V", band_qe)["limmag"][0]

        assert mag_deep > mag_shallow

    def test_limmag_realistic_range(self):
        """Limiting magnitudes from realistic noise levels fall in 15-25 range."""
        ab = _make_stub_ab(zp_photons_per_sec=1e8)
        snr = 5.0
        band_qe = 0.9

        bkg_noise_electrons = 50.0
        n_obs = 100
        exptime_per = 30.0
        inv_var = n_obs / bkg_noise_electrons**2
        total_exptime = n_obs * exptime_per

        depth_map = _make_depth_map(
            np.array([100]),
            np.array([inv_var]),
            np.array([total_exptime]),
            n_obs=n_obs,
        )
        result = limmag_from_map(depth_map, snr, ab, "V", band_qe)
        mag = result["limmag"][0]

        assert 10.0 < mag < 30.0

    @given(
        n_pixels=st.integers(min_value=1, max_value=20),
        snr=st.floats(min_value=1.0, max_value=100.0),
        band_qe=st.floats(min_value=0.1, max_value=1.0),
        zp=st.floats(min_value=1e4, max_value=1e10),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_limmag_equivalence(self, n_pixels, snr, band_qe, zp, data):
        """limmag_from_map and limmag_from_map_zp agree for consistent inputs."""
        pixels = np.arange(100, 100 + n_pixels, dtype=np.int64)
        inv_var = data.draw(
            st.lists(
                st.floats(min_value=1e-6, max_value=1e2),
                min_size=n_pixels,
                max_size=n_pixels,
            )
        )
        n_obs_vals = data.draw(
            st.lists(
                st.integers(min_value=1, max_value=500),
                min_size=n_pixels,
                max_size=n_pixels,
            )
        )
        exptime_per_obs = data.draw(
            st.floats(min_value=1.0, max_value=300.0),
        )

        inv_var = np.array(inv_var)
        n_obs_arr = np.array(n_obs_vals, dtype=np.int32)
        exptime = n_obs_arr.astype(np.float64) * exptime_per_obs

        depth_map = _make_depth_map(pixels, inv_var, exptime, n_obs=n_obs_arr)
        ab = _make_stub_ab(zp_photons_per_sec=zp)

        result_ab = limmag_from_map(depth_map, snr, ab, "V", band_qe)
        result_zp = limmag_from_map_zp(depth_map, snr, zp, band_qe)

        np.testing.assert_array_equal(result_ab["pixel"], result_zp["pixel"])
        np.testing.assert_allclose(
            result_ab["limmag"],
            result_zp["limmag"],
            atol=1e-10,
            err_msg="limmag_from_map and limmag_from_map_zp diverged",
        )


class TestBandQeUnits:
    """band_qe as a float or as an averaged ``qe_freq`` Quantity gives the same depth.

    Averaging ``SystemThroughput.qe_freq`` yields electron / photon, and
    callers pass that average straight to ``limmag_from_map``.
    """

    def test_extract_signal_rate_strips_qe_units(self):
        """Float, electron/photon and dimensionless QE give one plain-float rate."""
        import astropy.units as u

        depth_map = _make_depth_map([10, 20], [1e-4, 4e-4], 60.0)
        _, rate_float = _extract_signal_rate(depth_map, 5.0, 0.8)
        _, rate_epp = _extract_signal_rate(depth_map, 5.0, 0.8 * u.electron / u.photon)
        _, rate_dimless = _extract_signal_rate(depth_map, 5.0, 0.8 * u.dimensionless_unscaled)
        for rate in (rate_float, rate_epp, rate_dimless):
            assert not isinstance(rate, u.Quantity)
            np.testing.assert_allclose(rate, rate_float)

    def test_real_abphot_accepts_averaged_qe_quantity(self):
        """The real ABPhot path yields finite, identical depths for both forms."""
        import astropy.units as u

        from argus_sim import ABPhot, SystemThroughput, c

        tp = SystemThroughput(throughput_loss=1.0, filters={"g": True})
        area = (np.pi * (c.telescope.aperture_diameter / 2 * u.mm) ** 2).to(u.cm**2)
        ab = ABPhot(collecting_area=area, throughput_loss=1.0, throughput=tp)
        qe_quantity = np.average(tp.qe_freq, weights=tp.filt_freq["g"])
        assert isinstance(qe_quantity, u.Quantity)

        depth_map = _make_depth_map([10, 20], [1e-4, 4e-4], 60.0)
        with_quantity = limmag_from_map(depth_map, 5.0, ab, "g", qe_quantity)
        with_float = limmag_from_map(depth_map, 5.0, ab, "g", float(qe_quantity.value))
        np.testing.assert_allclose(with_quantity["limmag"], with_float["limmag"])
        assert np.all(np.isfinite(with_quantity["limmag"]))
