import astropy.time as atime
import astropy.units as u
import numpy as np
import pytest

from argus_sim import c
from argus_sim.survey import draw_seeing, sample_ratchets


class TestRatchetChronologicalOrder:
    """Verify that sample_ratchets() preserves chronological order."""

    @pytest.fixture()
    def ratchet_times_and_nights(self):
        ratchet_len_with_overhead = c.survey.ratchet_len + c.uptime.slew_len
        start = atime.Time("2024-06-01T00:00:00", scale="utc")
        end = atime.Time("2024-06-08T00:00:00", scale="utc")
        nratchets = int((end - start) / (ratchet_len_with_overhead * u.minute))
        times = atime.Time([start + i * ratchet_len_with_overhead * u.minute for i in range(nratchets)])
        times = times[times < end]
        nights = np.arange(len(times))
        return times, nights

    @pytest.mark.parametrize("fraction", [0.1, 0.25, 0.5, 0.75, 1.0])
    @pytest.mark.parametrize("seed", [0, 42, 12345])
    def test_sampled_ratchets_chronological(self, ratchet_times_and_nights, fraction, seed):
        times, nights = ratchet_times_and_nights
        rng = np.random.default_rng(seed)

        sampled_times, _ = sample_ratchets(times, nights, fraction, rng)
        if len(sampled_times) == 0:
            pytest.skip("zero-length sample")

        mjds = np.array([t.mjd for t in sampled_times])
        assert np.all(np.diff(mjds) >= 0), (
            f"Ratchet timestamps are not monotonically non-decreasing (seed={seed}, fraction={fraction})"
        )

    @pytest.mark.parametrize("seed", [0, 42, 99])
    def test_sample_count_matches_fraction(self, ratchet_times_and_nights, seed):
        times, nights = ratchet_times_and_nights
        rng = np.random.default_rng(seed)
        fraction = 0.5

        sampled_times, sampled_nights = sample_ratchets(times, nights, fraction, rng)
        expected = int(np.round(len(times) * fraction))
        assert len(sampled_times) == expected
        assert len(sampled_nights) == expected


class TestSeeingDistribution:
    @pytest.mark.parametrize("seed", [42, 123, 9999])
    def test_lognormal_median_converges_to_seeing_mean(self, seed):
        seeing_mean = c.observatory.seeing_mean
        seeing_std = c.observatory.seeing_std

        rng = np.random.default_rng(seed)
        samples = np.array([draw_seeing(seeing_mean, seeing_std, rng) for _ in range(10_000)])

        median = np.median(samples)
        assert median == pytest.approx(seeing_mean, rel=0.05), (
            f"Sample median {median:.4f} does not converge to seeing_mean {seeing_mean} within 5%"
        )

    def test_draw_seeing_positive(self):
        rng = np.random.default_rng(0)
        for _ in range(1000):
            s = draw_seeing(2.0, 0.3, rng)
            assert s > 0


def test_write_parquet_with_metadata_round_trip(tmp_path):
    import pandas as pd
    import pyarrow.parquet as pq

    from argus_sim.survey import write_parquet_with_metadata

    df = pd.DataFrame({"healpix": [1, 2], "band": ["g", "rho"], "limmag": [20.1, 19.7]})
    path = tmp_path / "ratchet.parquet"
    write_parquet_with_metadata(df, str(path), {"epoch_start_mjd": "61772.07", "n_epochs": 15})
    table = pq.read_table(path)
    assert table.schema.metadata[b"epoch_start_mjd"] == b"61772.07"
    assert table.schema.metadata[b"n_epochs"] == b"15"
    pd.testing.assert_frame_equal(table.to_pandas(), df)
