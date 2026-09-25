"""On-sky depth: extinction enters as a per-exposure transmission weight.

Properties: with T = 1 the on-sky sum is the accumulator's sum; with a
constant transmission T the on-sky depth is exactly k X = -2.5 log10 T
shallower; the above-atmosphere sum from rows equals what
``CadenceAccumulator.ingest`` writes for the same rows.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from argus_sim.observatory import pickering_airmass
from argus_sim.onsky_depth import (
    accumulate_onsky,
    atmospheric_transmission,
    extinction_coefficient,
    limmag_from_inv_sum,
)


def _stub_ab(zp=1e6):
    ab = MagicMock()
    ab.mag_from_photons = lambda _b, photons: -2.5 * np.log10(np.asarray(photons, dtype=np.float64) / zp)
    return ab


class TestAccumulateOnsky:
    """Row accumulation and the extinction shift it implies."""

    def test_unit_transmission_matches_above_atmosphere(self):
        """T = 1 everywhere makes the two sums identical."""
        hpx = np.array([5, 5, 9])
        out = accumulate_onsky(hpx, [10.0, 12.0, 8.0], [2.0, 3.0, 1.0], np.ones(3), n_epochs=4, exptime_s=60.0)
        np.testing.assert_allclose(out["inv_onsky"], out["inv_above"])
        np.testing.assert_array_equal(out["n_obs"], [8, 4])
        np.testing.assert_allclose(out["exptime_s"], [480.0, 240.0])

    def test_rows_without_background_variance_are_dropped(self):
        """A row whose noise is all source shot contributes nothing, as in ingest."""
        out = accumulate_onsky([1, 2], [5.0, 3.0], [5.0, 1.0], [0.9, 0.9], n_epochs=1, exptime_s=60.0)
        np.testing.assert_array_equal(out["pixel"], [2])

    @settings(max_examples=50, deadline=None)
    @given(
        airmass=st.floats(min_value=1.0, max_value=3.0),
        atm_tp_am1=st.floats(min_value=0.5, max_value=0.99),
        n_epochs=st.integers(min_value=1, max_value=30),
    )
    def test_constant_airmass_shifts_depth_by_k_times_airmass(self, airmass, atm_tp_am1, n_epochs):
        """Constant transmission T: on-sky limmag = above-atmosphere limmag - (-2.5 log10 T)."""
        alt = np.degrees(np.arcsin(1.0 / airmass))
        trans = atmospheric_transmission(np.full(3, alt), atm_tp_am1)
        out = accumulate_onsky([7, 7, 7], [10.0, 11.0, 9.0], [1.0, 2.0, 0.5], trans, n_epochs, 60.0)
        ab = _stub_ab()
        m_above = limmag_from_inv_sum(out["inv_above"], out["n_obs"], out["exptime_s"], 5.0, ab, "g", 0.8)
        m_onsky = limmag_from_inv_sum(out["inv_onsky"], out["n_obs"], out["exptime_s"], 5.0, ab, "g", 0.8)
        expected = -2.5 * np.log10(float(trans[0]))
        assert expected == pytest.approx(extinction_coefficient(atm_tp_am1) * float(pickering(alt)), rel=1e-9)
        np.testing.assert_allclose(m_above - m_onsky, expected, rtol=1e-9)

    def test_above_sum_matches_cadence_accumulator(self, tmp_path):
        """The rows path and CadenceAccumulator.ingest agree on inv_bkg_noise_sq_sum."""
        from argus_sim.cadence_accumulator import CadenceAccumulator, build_window_registry

        hpx = np.array([100, 100, 200, 300])
        noise = np.array([10.0, 12.0, 20.0, 15.0])
        shot = np.array([3.0, 2.0, 4.0, 5.0])
        registry = build_window_registry([0], [60000.1], [0], ["night"])
        acc = CadenceAccumulator(registry, ["night"], 128, str(tmp_path), mjds={0: 60000.1})
        acc.ingest(
            0,
            {"healpix": hpx, "noise": noise, "source_shot": shot, "exptime_s": 60.0, "band": np.array(["g"] * 4)},
            n_epochs=3,
        )
        manifest = acc.finalize()
        import healsparse as hsp

        f = tmp_path / manifest["cadence_levels"]["night"]["windows"][0]["bands"][0]["file"]
        m = hsp.HealSparseMap.read(str(f))
        out = accumulate_onsky(hpx, noise, shot, np.full(4, 0.8), n_epochs=3, exptime_s=60.0)
        vals = m.get_values_pix(out["pixel"])
        np.testing.assert_allclose(vals["inv_bkg_noise_sq_sum"], out["inv_above"], rtol=1e-12)
        np.testing.assert_array_equal(vals["n_obs"], out["n_obs"])


def pickering(alt):
    """Pickering airmass, imported lazily to keep the module header light."""
    from argus_sim.observatory import pickering_airmass

    return pickering_airmass(alt)


class TestFullSystem:
    """Reported depths are through the full system (optics x atmosphere at each airmass)."""

    def test_system_transmission_is_the_exact_efficiency_at_the_row_airmass(self):
        from argus_sim.onsky_depth import system_transmission
        from argus_sim.spectral_sky import signal_throughput

        X = np.linspace(0.0, 40.0, 801)
        bp = {"signal_tp_table": (X, 0.8 * 0.75**X)}
        alt = np.array([90.0, 60.0, 40.0])
        np.testing.assert_allclose(system_transmission(alt, bp), 0.8 * atmospheric_transmission(alt, 0.75), rtol=1e-12)
        np.testing.assert_allclose(system_transmission(alt, bp), signal_throughput(bp, pickering_airmass(alt)))

    def test_accumulator_weights_by_signal_throughput(self, tmp_path):
        from argus_sim.cadence_accumulator import CadenceAccumulator, build_window_registry

        ratchet_nums = np.array([0])
        mjds = np.array([60000.1])
        levels = ["night"]
        registry = build_window_registry(ratchet_nums, mjds, np.array([0]), levels)
        acc = CadenceAccumulator(registry, levels, nside_sparse=64, output_dir=str(tmp_path), mjds={0: 60000.1})
        noise = np.array([10.0, 12.0])
        shot = np.array([2.0, 3.0])
        tp = np.array([0.6, 0.5])
        acc.ingest(
            0,
            {
                "healpix": np.array([5, 9]),
                "noise": noise,
                "source_shot": shot,
                "exptime_s": 60.0,
                "band": np.array(["g", "g"]),
                "signal_tp": tp,
            },
            n_epochs=14,
        )
        manifest = acc.finalize()
        import healsparse as hsp

        f = manifest["cadence_levels"]["night"]["windows"][0]["bands"][0]["file"]
        m = hsp.HealSparseMap.read(str(tmp_path / f))
        vals = m.get_values_pix(np.array([5, 9]))
        np.testing.assert_allclose(vals["inv_bkg_noise_sq_sum"], 14 * tp**2 / (noise**2 - shot**2))
