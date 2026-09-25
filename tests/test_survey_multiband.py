import astropy.time as atime
import numpy as np
import pytest

from argus_sim import CradlePointing, c


class TestCradleFilterMaps:
    @pytest.fixture(scope="class")
    def cradle(self):
        return CradlePointing()

    def test_ring_cradle_has_filter_maps(self, cradle):
        assert hasattr(cradle, "filter_maps")
        assert cradle.filter_maps is not None
        assert len(cradle.filter_maps) == len(set(c.filter_strategy.options))

    def test_grid_cradle_has_no_filter_maps(self):
        cradle = CradlePointing(tiling="grid", n_telescopes=650)
        assert cradle.filter_maps is None

    def test_hpx_table_has_filter_column(self, cradle):
        t = atime.Time("2024-06-15T03:00:00", scale="utc")
        table = cradle.to_hpx_table_at(t)
        assert "filter" in table.colnames

    def test_filter_values_are_valid(self, cradle):
        t = atime.Time("2024-06-15T03:00:00", scale="utc")
        table = cradle.to_hpx_table_at(t)
        valid = set(c.filter_strategy.options) | {""}
        for f in table["filter"]:
            assert f in valid

    def test_multiple_bands_present(self, cradle):
        t = atime.Time("2024-06-15T03:00:00", scale="utc")
        table = cradle.to_hpx_table_at(t)
        bands_present = set(table["filter"])
        bands_present.discard("")
        assert len(bands_present) > 1, "Expected multiple filter bands in healpix table"


class TestCradlePixelBandRows:
    """Overlap pixels are imaged in every covering band at once, so the
    healpix table must carry one row per (pixel, band) and the band share
    of rows must follow the camera counts rather than a tie-break.
    """

    @pytest.fixture(scope="class")
    def table(self):
        return CradlePointing().to_hpx_table_at(atime.Time("2024-06-15T03:00:00", scale="utc"))

    def test_pixel_band_pairs_unique(self, table):
        pairs = list(zip(table["healpix"], table["filter"]))
        assert len(pairs) == len(set(pairs))

    def test_overlap_pixels_have_one_row_per_band(self, table):
        _, per_pixel = np.unique(table["healpix"], return_counts=True)
        n_bands = len(set(c.filter_strategy.options))
        assert per_pixel.max() == n_bands
        assert np.mean(per_pixel > 1) > 0.05, "strip overlaps should cover a few percent of pixels"

    def test_no_unfiltered_rows(self, table):
        assert np.all(table["filter"] != "")

    def test_uncovered_footprint_cells_are_an_edge_rind(self):
        """Cells of the footprint MOC outside every filter map are partially
        covered edge cells: a few percent of the table, and nearly all within
        two cells of a covered cell -- a rind, not holes in the footprint."""
        import healpy as hp

        cradle = CradlePointing()
        tab = cradle.hpx_pointings
        az, alt, pix = np.asarray(tab["az"]), np.asarray(tab["alt"]), np.asarray(tab["hpx"]).astype(int)
        nside = 2**c.survey.sim_depth
        covered = np.zeros(len(pix), dtype=bool)
        for hsmap in cradle.filter_maps.values():
            covered |= hsmap.get_values_pos(az, alt, lonlat=True) > 0
        uncovered = ~covered
        assert 0.0 < uncovered.mean() < 0.08
        covered_set = set(pix[covered].tolist())

        def touches(cells):
            return np.array([any(n in covered_set for n in row if n >= 0) for row in cells])

        ring1 = hp.get_all_neighbours(nside, pix[uncovered], nest=True).T
        near = touches(ring1)
        assert near.mean() > 0.9
        far_rows = ring1[~near]
        within_two = np.array(
            [
                any(touches(hp.get_all_neighbours(nside, n, nest=True)[None, :])[0] for n in row if n >= 0)
                for row in far_rows
            ]
        )
        assert (near.sum() + within_two.sum()) / uncovered.sum() > 0.99

    def test_band_row_share_tracks_camera_share(self, table):
        rows = table
        shares = {
            b: c.filter_strategy.options.count(b) / len(c.filter_strategy.options)
            for b in set(c.filter_strategy.options)
        }
        for band, share in shares.items():
            row_share = float(np.mean(rows["filter"] == band))
            assert row_share == pytest.approx(share, abs=0.03), f"{band}: rows {row_share:.3f} vs strategy {share:.3f}"


class TestMultiBandParams:
    def test_band_params_built_for_all_filters(self):
        from argus_sim.survey import Survey

        survey = Survey(with_cradle=False)
        from argus_sim.spectral_sky import SpectralSky

        band_params = {}
        for band in c.filter_strategy.options:
            signal_tp = np.average(
                survey.throughputs.optics[band] * survey.throughputs.filt_freq.atmosphere,
                weights=survey.throughputs.filt_freq[band],
            )
            spectral_sky = SpectralSky(survey.throughputs, survey.ab, band)
            band_qe = np.average(survey.throughputs.qe_freq, weights=survey.throughputs.filt_freq[band])
            band_params[band] = {
                "signal_tp": signal_tp,
                "band_qe": band_qe,
                "sky": spectral_sky,
            }

        for band in c.filter_strategy.options:
            assert band in band_params
            assert band_params[band]["signal_tp"] > 0
            assert band_params[band]["band_qe"] > 0

    def test_different_bands_have_different_throughputs(self):
        from argus_sim.survey import Survey

        survey = Survey(with_cradle=False)
        throughputs = {}
        for band in c.filter_strategy.options:
            throughputs[band] = np.average(
                survey.throughputs.optics[band] * survey.throughputs.filt_freq.atmosphere,
                weights=survey.throughputs.filt_freq[band],
            )
        values = list(throughputs.values())
        assert len(set(round(v, 6) for v in values)) > 1, "All bands have identical throughput"
