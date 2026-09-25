"""Tests for forward Monte Carlo uncertainty propagation."""

import numpy as np
import pytest

from argus_sim.config import Config
import astropy.units as u

from argus_sim.forward_mc import (
    DEFAULT_CADENCE_WINDOWS,
    DEFAULT_PRIORS,
    DEFAULT_RATCHET_PRIORS,
    DEFAULT_SAMPLE_PRIORS,
    LUNAR_MODES,
    MONTHLY_WEATHER_STATS,
    DepthResult,
    ForwardMC,
    Prior,
    SampleResult,
    _annual_mean_clear_fraction,
    _get_config_field,
    _n_eff_for_cadence,
    _set_config_field,
    _weighted_percentile,
    build_ratchet_grid,
    compute_grid_weights,
    compute_sensitivity,
    draw_correlated_atmosphere,
    draw_monthly_weather,
    evaluate_single_image_depth,
    perturb_config,
    plot_priors,
    plot_sensitivity,
)
from argus_sim.spectral_sky import build_band_params


class TestPrior:
    def test_normal_draw_is_finite(self):
        p = Prior("telescope.readnoise", "normal", {"mean": 1.4, "std": 0.2})
        rng = np.random.default_rng(42)
        for _ in range(100):
            v = p.draw(rng)
            assert np.isfinite(v)

    def test_lognormal_draw_is_positive(self):
        p = Prior("observatory.seeing_mean", "lognormal", {"mean": 0.0, "std": 0.15})
        rng = np.random.default_rng(42)
        for _ in range(100):
            assert p.draw(rng) > 0

    def test_uniform_draw_in_bounds(self):
        p = Prior("telescope.readnoise", "uniform", {"low": 1.5, "high": 2.5})
        rng = np.random.default_rng(42)
        for _ in range(100):
            v = p.draw(rng)
            assert 1.5 <= v <= 2.5

    def test_invgamma_draw_is_positive(self):
        p = Prior("telescope.dark_current", "invgamma", {"a": 2.629, "loc": -0.0003, "scale": 0.002416})
        rng = np.random.default_rng(42)
        values = [p.draw(rng) for _ in range(500)]
        assert all(np.isfinite(v) for v in values)
        assert all(v > -0.001 for v in values)

    def test_burr12_draw_is_positive(self):
        p = Prior("telescope.readnoise", "burr12", {"c": 6.275, "d": 0.2217, "loc": 0.9082, "scale": 0.1205})
        rng = np.random.default_rng(42)
        values = [p.draw(rng) for _ in range(500)]
        assert all(np.isfinite(v) for v in values)
        assert all(v > 0 for v in values)

    def test_clip_lower_bound(self):
        p = Prior("telescope.readnoise", "normal", {"mean": 0.1, "std": 0.5}, clip=(0.0, None))
        rng = np.random.default_rng(42)
        for _ in range(1000):
            assert p.draw(rng) >= 0.0

    def test_clip_both_bounds(self):
        p = Prior("x", "normal", {"mean": 0.5, "std": 1.0}, clip=(0.0, 1.0))
        rng = np.random.default_rng(42)
        for _ in range(1000):
            v = p.draw(rng)
            assert 0.0 <= v <= 1.0

    def test_unknown_dist_raises(self):
        p = Prior("x", "beta", {"a": 1, "b": 1})
        with pytest.raises(ValueError, match="Unknown distribution"):
            p.draw(np.random.default_rng(0))


class TestConfigFieldAccess:
    def test_get_nested_field(self):
        cfg = Config(sim_version="test")
        assert _get_config_field(cfg, "telescope.readnoise") == cfg.telescope.readnoise

    def test_set_nested_field(self):
        cfg = Config(sim_version="test")
        _set_config_field(cfg, "telescope.readnoise", 99.0)
        assert cfg.telescope.readnoise == 99.0

    def test_set_does_not_affect_other_fields(self):
        cfg = Config(sim_version="test")
        original_seeing = cfg.observatory.seeing_mean
        _set_config_field(cfg, "telescope.readnoise", 99.0)
        assert cfg.observatory.seeing_mean == original_seeing


class TestPerturbConfig:
    def test_returns_independent_copy(self):
        base = Config(sim_version="test")
        priors = [Prior("telescope.readnoise", "normal", {"mean": 1.4, "std": 0.2})]
        rng = np.random.default_rng(42)

        perturbed, _ = perturb_config(base, priors, rng)

        assert perturbed is not base
        assert perturbed.telescope is not base.telescope
        assert base.telescope.readnoise == 1.2  # unchanged

    def test_drawn_values_match_theta(self):
        base = Config(sim_version="test")
        priors = [
            Prior("telescope.readnoise", "normal", {"mean": 1.4, "std": 0.2}),
            Prior("observatory.seeing_mean", "lognormal", {"mean": 0.0, "std": 0.15}),
        ]
        rng = np.random.default_rng(42)
        perturbed, theta = perturb_config(base, priors, rng)

        for field, expected in theta.items():
            actual = _get_config_field(perturbed, field)
            assert actual == pytest.approx(expected)

    def test_all_default_sample_priors_apply_to_config(self):
        base = Config(sim_version="test")
        rng = np.random.default_rng(42)
        perturbed, theta = perturb_config(base, DEFAULT_SAMPLE_PRIORS, rng)

        for prior in DEFAULT_SAMPLE_PRIORS:
            assert prior.field in theta
            assert _get_config_field(perturbed, prior.field) == pytest.approx(theta[prior.field])

    def test_reproducibility_with_same_seed(self):
        base = Config(sim_version="test")
        priors = DEFAULT_SAMPLE_PRIORS

        _, theta1 = perturb_config(base, priors, np.random.default_rng(123))
        _, theta2 = perturb_config(base, priors, np.random.default_rng(123))

        for k in theta1:
            assert theta1[k] == theta2[k]

    def test_weather_fraction_prior_clipped(self):
        base = Config(sim_version="test")
        priors = [Prior("telescope.throughput_loss", "normal", {"mean": 0.87, "std": 0.05}, clip=(0.5, 1.0))]
        rng = np.random.default_rng(42)
        for _ in range(100):
            perturbed, theta = perturb_config(base, priors, rng)
            assert 0.5 <= theta["telescope.throughput_loss"] <= 1.0
            assert perturbed.telescope.throughput_loss == theta["telescope.throughput_loss"]

    def test_transparency_prior(self):
        base = Config(sim_version="test")
        priors = [Prior("observatory.transparency", "normal", {"mean": 0.75, "std": 0.03}, clip=(0.5, 1.0))]
        rng = np.random.default_rng(42)
        for _ in range(100):
            _, theta = perturb_config(base, priors, rng)
            assert 0.5 <= theta["observatory.transparency"] <= 1.0


class TestRatchetGrid:
    def test_base_grid_excludes_bright(self):
        grid = build_ratchet_grid(lunar_mode="base")
        for pt in grid:
            assert pt.lunar_illumination < 0.93

    def test_base_grid_covers_all_seasons(self):
        grid = build_ratchet_grid(lunar_mode="base")
        seasons = {pt.season for pt in grid}
        assert seasons == {"winter", "equinox", "summer"}

    def test_base_grid_covers_all_airmasses(self):
        grid = build_ratchet_grid(lunar_mode="base")
        airmasses = {pt.airmass for pt in grid}
        assert airmasses == {1.0, 1.2, 1.6}

    def test_dark_grid_only_new_moon(self):
        grid = build_ratchet_grid(lunar_mode="dark")
        assert len(grid) == 9  # 1 phase × 3 seasons × 3 airmasses
        for pt in grid:
            assert pt.lunar_offset_days == 0

    def test_bright_grid_only_high_illumination(self):
        grid = build_ratchet_grid(lunar_mode="bright")
        for pt in grid:
            assert pt.lunar_illumination >= 0.93

    def test_all_modes_cover_all_seasons_and_airmasses(self):
        for mode in LUNAR_MODES:
            grid = build_ratchet_grid(lunar_mode=mode)
            assert len(grid) > 0, f"Empty grid for mode={mode}"
            seasons = {pt.season for pt in grid}
            assert seasons == {"winter", "equinox", "summer"}, f"Missing seasons for mode={mode}"
            airmasses = {pt.airmass for pt in grid}
            assert airmasses == {1.0, 1.2, 1.6}, f"Missing airmasses for mode={mode}"

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown lunar_mode"):
            build_ratchet_grid(lunar_mode="twilight")

    def test_weights_sum_to_one(self):
        for mode in LUNAR_MODES:
            grid = build_ratchet_grid(lunar_mode=mode)
            weights = compute_grid_weights(grid)
            assert weights.sum() == pytest.approx(1.0), f"Weights don't sum to 1 for mode={mode}"

    def test_weights_shape_matches_grid(self):
        grid = build_ratchet_grid(lunar_mode="base")
        weights = compute_grid_weights(grid)
        assert len(weights) == len(grid)

    def test_winter_has_higher_weight_than_summer(self):
        grid = build_ratchet_grid(lunar_mode="base")
        weights = compute_grid_weights(grid)
        winter_w = sum(w for pt, w in zip(grid, weights) if pt.season == "winter")
        summer_w = sum(w for pt, w in zip(grid, weights) if pt.season == "summer")
        assert winter_w > summer_w

    def test_grid_points_have_illumination(self):
        grid = build_ratchet_grid(lunar_mode="base")
        for pt in grid:
            assert 0.0 <= pt.lunar_illumination <= 1.0


class TestCorrelatedAtmosphere:
    def test_seeing_is_positive(self):
        rng = np.random.default_rng(42)
        for _ in range(1000):
            seeing, _ = draw_correlated_atmosphere(rng, 0.87, 0.15, 0.75, 0.03)
            assert seeing > 0

    def test_transparency_in_bounds(self):
        rng = np.random.default_rng(42)
        for _ in range(1000):
            _, transparency = draw_correlated_atmosphere(rng, 0.87, 0.15, 0.75, 0.03)
            assert 0.5 <= transparency <= 1.0

    def test_correlation_sign(self):
        rng = np.random.default_rng(42)
        seeings = []
        transparencies = []
        for _ in range(5000):
            s, t = draw_correlated_atmosphere(rng, 0.87, 0.15, 0.75, 0.03, rho=0.8)
            seeings.append(s)
            transparencies.append(t)
        corr = np.corrcoef(seeings, transparencies)[0, 1]
        # Worse seeing (larger) should correlate with worse transparency (lower)
        assert corr < 0

    def test_zero_correlation(self):
        rng = np.random.default_rng(42)
        seeings = []
        transparencies = []
        for _ in range(10000):
            s, t = draw_correlated_atmosphere(rng, 0.87, 0.15, 0.75, 0.03, rho=0.0)
            seeings.append(s)
            transparencies.append(t)
        corr = np.corrcoef(seeings, transparencies)[0, 1]
        assert abs(corr) < 0.05


class TestMonthlyWeather:
    def test_draw_returns_12_months(self):
        rng = np.random.default_rng(42)
        weather = draw_monthly_weather(rng)
        assert len(weather) == 12
        assert set(weather.keys()) == set(range(1, 13))

    def test_values_in_unit_interval(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            weather = draw_monthly_weather(rng)
            for month, f in weather.items():
                assert 0.0 <= f <= 1.0, f"Month {month} out of range: {f}"

    def test_reproducibility(self):
        w1 = draw_monthly_weather(np.random.default_rng(123))
        w2 = draw_monthly_weather(np.random.default_rng(123))
        for m in range(1, 13):
            assert w1[m] == w2[m]

    def test_stats_table_populated(self):
        assert len(MONTHLY_WEATHER_STATS) == 12
        for month in range(1, 13):
            mu, sigma = MONTHLY_WEATHER_STATS[month]
            assert 0.0 < mu < 1.0
            assert sigma >= 0.0

    def test_annual_mean_in_range(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            weather = draw_monthly_weather(rng)
            f_annual = _annual_mean_clear_fraction(weather)
            assert 0.0 <= f_annual <= 1.0


class TestWeightedPercentile:
    def test_uniform_weights_match_numpy(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.ones(5)
        assert _weighted_percentile(values, weights, 50) == pytest.approx(np.median(values), abs=0.5)

    def test_dominant_weight_shifts_percentile(self):
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.01, 0.01, 0.98])
        p90 = _weighted_percentile(values, weights, 90)
        # 90th percentile should be near 3.0 when 98% of weight is on 3.0
        assert p90 > 2.8


class TestForwardMC:
    def test_init_stores_priors(self):
        cfg = Config(sim_version="test")
        mc = ForwardMC(cfg)
        assert len(mc.priors) == len(DEFAULT_PRIORS)

    def test_custom_priors(self):
        cfg = Config(sim_version="test")
        custom = [Prior("telescope.readnoise", "normal", {"mean": 1.0, "std": 0.1})]
        mc = ForwardMC(cfg, priors=custom)
        assert len(mc.priors) == 1

    def test_cradle_caching(self):
        cfg = Config(sim_version="test")
        mc = ForwardMC(cfg)
        assert mc._cradle is None

    def test_to_dataframe_structure(self):
        cfg = Config(sim_version="test")
        mc = ForwardMC(cfg)
        mc.results = [
            SampleResult(
                sample_id=0,
                theta={"telescope.readnoise": 1.3, "observatory.seeing_mean": 0.9},
                ratchet_depths={"g": [20.5, 20.3, 20.7]},
                median_depth={"g": 20.5},
                depth_10pct={"g": 19.8},
                depth_90pct={"g": 21.0},
            ),
        ]
        df = mc.to_dataframe()
        assert len(df) == 1
        assert "theta_telescope.readnoise" in df.columns
        assert "median_depth_g" in df.columns

    def test_default_priors_count(self):
        assert len(DEFAULT_SAMPLE_PRIORS) == 4
        assert len(DEFAULT_RATCHET_PRIORS) == 2
        assert len(DEFAULT_PRIORS) == 6
        fields = {p.field for p in DEFAULT_PRIORS}
        assert "telescope.readnoise" in fields
        assert "observatory.transparency" in fields

    def test_degenerate_zero_std_deterministic(self, tmp_path):
        """Zero-uncertainty priors produce deterministic depths matching direct evaluation."""
        from argus_sim.ab_system import ABPhot
        from argus_sim.observatory import Observatory
        from argus_sim.photon_budget import NoiseBudget
        from argus_sim.psf import GaussianPSF
        from argus_sim.throughput import SystemThroughput

        cfg = Config(sim_version="test")
        cfg.telescope.psf_model = "gaussian"  # deterministic PSF; the delivered default draws pixel phases
        cfg.observatory.seeing_std = 0.0
        cfg.observatory.seasonal_seeing = False  # the seasonal table carries its own spread; this test wants zero

        zero_ratchet_priors = [
            Prior("observatory.seeing_mean", "normal", {"mean": cfg.observatory.seeing_mean, "std": 0.0}),
            Prior("observatory.transparency", "normal", {"mean": cfg.observatory.transparency, "std": 0.0}),
        ]

        mc = ForwardMC(
            cfg,
            sample_priors=[],
            ratchet_priors=zero_ratchet_priors,
            output_dir=str(tmp_path),
        )
        results = mc.run(n_samples=2, seed=42)

        # Both samples must produce identical per-ratchet depths
        for band in results[0].ratchet_depths:
            for d0, d1 in zip(results[0].ratchet_depths[band], results[1].ratchet_depths[band]):
                assert d0 == pytest.approx(d1), f"Depths differ for {band}"
            assert results[0].median_depth[band] == pytest.approx(results[1].median_depth[band])

        # Direct evaluation of first grid point must match MC ratchet_depths[0]
        obs = Observatory(config=cfg)
        tp_loss = cfg.telescope.throughput_loss
        throughputs = SystemThroughput(throughput_loss=tp_loss)
        ab = ABPhot(collecting_area=cfg.telescope.collecting_area, throughput_loss=tp_loss, throughput=throughputs)
        noise_budget = NoiseBudget(
            read_noise=cfg.telescope.readnoise * u.electron,
            dark_current=cfg.telescope.dark_current * u.electron / u.second,
        )
        psf_model = GaussianPSF(
            stamp_size=cfg.survey.stamp_size,
            stamp_resolution=cfg.survey.stamp_resolution,
            plate_scale=cfg.telescope.plate_scale,
            pixel_size=cfg.telescope.pixel_size,
            rms_spot_size=cfg.telescope.spot_size,
        )
        band_params = build_band_params(set(cfg.filter_strategy.options), throughputs, ab)
        grid = build_ratchet_grid()

        dr = evaluate_single_image_depth(
            obs=obs,
            noise_budget=noise_budget,
            psf_model=psf_model,
            ab=ab,
            band_params=band_params,
            grid_point=grid[0],
            seeing=cfg.observatory.seeing_mean,
            transparency=cfg.observatory.transparency,
            pixel_phase=None,
            cfg=cfg,
        )

        for band in dr.single_depths:
            assert dr.single_depths[band] == pytest.approx(results[0].ratchet_depths[band][0])
            assert dr.psf_fwhm[band] > 0
            assert dr.optical_psf_size[band] > 0


class TestSampleResult:
    def test_dataclass_fields(self):
        r = SampleResult(
            sample_id=0,
            theta={"a": 1.0},
            ratchet_depths={"b": [2.0, 2.1]},
            median_depth={"b": 2.0},
            depth_10pct={"b": 1.5},
            depth_90pct={"b": 2.5},
        )
        assert r.sample_id == 0
        assert r.theta["a"] == 1.0
        assert len(r.ratchet_depths["b"]) == 2


class TestCadenceNEff:
    """Unit tests for the N_eff cadence scaling helper."""

    def test_single_exposure_always_one(self):
        assert _n_eff_for_cadence("1min", "winter", 1.0, 60.0) == 1.0
        assert _n_eff_for_cadence("1min", "summer", 0.5, 60.0) == 1.0

    def test_one_hour_ignores_weather(self):
        n = _n_eff_for_cadence("1hour", "winter", 0.5, 60.0)
        assert n == 60.0

    def test_one_night_uses_season_hours(self):
        n_w = _n_eff_for_cadence("1night", "winter", 1.0, 60.0)
        n_s = _n_eff_for_cadence("1night", "summer", 1.0, 60.0)
        assert n_w == 11.0 * 60
        assert n_s == 6.0 * 60
        assert n_w > n_s

    def test_multinight_scales_with_weather(self):
        n_full = _n_eff_for_cadence("1week", "winter", 1.0, 60.0)
        n_half = _n_eff_for_cadence("1week", "winter", 0.5, 60.0)
        assert n_half == pytest.approx(n_full * 0.5)

    def test_six_month_scales_with_weather(self):
        n_full = _n_eff_for_cadence("6month", "equinox", 1.0, 60.0)
        n_half = _n_eff_for_cadence("6month", "equinox", 0.5, 60.0)
        assert n_half == pytest.approx(n_full * 0.5)

    def test_five_year_uses_weather(self):
        n_5y_full = _n_eff_for_cadence("5year", "equinox", 1.0, 60.0)
        n_5y_half = _n_eff_for_cadence("5year", "equinox", 0.5, 60.0)
        assert n_5y_half == pytest.approx(n_5y_full * 0.5)

    def test_unknown_window_raises(self):
        with pytest.raises(ValueError, match="Unknown cadence"):
            _n_eff_for_cadence("10years", "winter", 1.0, 60.0)

    @pytest.mark.parametrize("window", ["1hour", "1night", "1week", "6month", "5year"])
    def test_band_share_scales_multi_exposure_windows(self, window):
        full = _n_eff_for_cadence(window, "equinox", 0.7, 60.0)
        half = _n_eff_for_cadence(window, "equinox", 0.7, 60.0, band_share=0.5)
        assert half == pytest.approx(0.5 * full)

    @pytest.mark.parametrize("window,exptime", [("1sec", 1.0), ("1sec", 60.0), ("1min", 60.0)])
    def test_band_share_never_splits_a_single_exposure(self, window, exptime):
        assert _n_eff_for_cadence(window, "winter", 1.0, exptime, band_share=0.5) == 1.0

    def test_band_share_floor_is_one_exposure(self):
        assert _n_eff_for_cadence("1min", "winter", 1.0, 30.0, band_share=0.1) == 1.0

    @pytest.mark.parametrize("exptime", [60.0, 120.0])
    def test_one_second_window_is_one_exposure_at_long_exptime(self, exptime):
        """cadence_1sec equals cadence_1min whenever the exposure is >= 60 s."""
        assert _n_eff_for_cadence("1sec", "equinox", 1.0, exptime) == 1.0
        assert _n_eff_for_cadence("1min", "equinox", 1.0, exptime) == 1.0

    def test_one_second_window_differs_from_one_minute_in_bright_mode(self):
        assert _n_eff_for_cadence("1min", "equinox", 1.0, 1.0) == 60.0

    def test_default_share_is_unchanged(self):
        for w in DEFAULT_CADENCE_WINDOWS:
            assert _n_eff_for_cadence(w, "summer", 0.6, 60.0) == _n_eff_for_cadence(w, "summer", 0.6, 60.0, 1.0)


class TestBandOrderDeterminism:
    """Band iteration order must not depend on set hashing, or the RNG stream
    consumed per band differs between processes and the same seed gives
    different depths.
    """

    def test_band_params_keys_sorted_regardless_of_option_order(self):
        from argus_sim import ABPhot, SystemThroughput, c
        from argus_sim.spectral_sky import build_band_params

        tp = SystemThroughput(throughput_loss=1.0)
        area = (np.pi * (c.telescope.aperture_diameter / 2 * u.mm) ** 2).to(u.cm**2)
        ab = ABPhot(collecting_area=area, throughput_loss=1.0, throughput=tp)
        for options in (["g", "r+i", "g", "r+i"], ["r+i", "g", "r+i", "g"]):
            keys = list(build_band_params(sorted(set(options)), tp, ab).keys())
            assert keys == sorted(keys)

    def test_same_seed_same_depths_across_option_order(self, tmp_path):
        out = []
        for options in (["g", "r+i", "g", "r+i"], ["r+i", "g", "r+i", "g"]):
            cfg = Config(sim_version="test")
            cfg.telescope.psf_model = "gaussian"  # deterministic PSF; the delivered default draws pixel phases
            cfg.filter_strategy.options = options
            mc = ForwardMC(cfg, sample_priors=[], output_dir=str(tmp_path / options[0]), cadence_windows=("1night",))
            out.append(mc.run(n_samples=1, seed=5)[0])
        assert out[0].median_depth == out[1].median_depth
        assert out[0].cadence_depths == out[1].cadence_depths


class TestBandShares:
    @pytest.fixture(autouse=True)
    def _import(self):
        from argus_sim.forward_mc import band_shares

        self.band_shares = band_shares

    def test_one_to_one(self):
        assert self.band_shares(["g", "rho", "g", "rho"]) == {"g": 0.5, "rho": 0.5}

    def test_two_to_one(self):
        shares = self.band_shares(["g", "g", "rho"])
        assert shares["g"] == pytest.approx(2 / 3)
        assert shares["rho"] == pytest.approx(1 / 3)

    def test_single_band(self):
        assert self.band_shares(["g"]) == {"g": 1.0}

    def test_shares_sum_to_one(self):
        assert sum(self.band_shares(["g", "r+i", "i", "g"]).values()) == pytest.approx(1.0)

    def test_monotonically_increasing(self):
        """With 1s exposures all windows are strictly increasing."""
        windows = DEFAULT_CADENCE_WINDOWS
        n_effs = [_n_eff_for_cadence(w, "equinox", 0.9, 1.0) for w in windows]
        for i in range(len(n_effs) - 1):
            assert n_effs[i] < n_effs[i + 1], f"{windows[i]} >= {windows[i + 1]}"


class TestCadenceDepths:
    """Integration tests for cadence depth evaluation."""

    def test_cadence_depths_populated_in_mc_run(self, tmp_path):
        cfg = Config(sim_version="test")
        mc = ForwardMC(
            cfg,
            sample_priors=[],
            output_dir=str(tmp_path),
            cadence_windows=("1min", "1hour", "1night"),
        )
        results = mc.run(n_samples=1, seed=42)
        r = results[0]

        assert len(r.cadence_depths) == 3
        for window in ("1min", "1hour", "1night"):
            assert window in r.cadence_depths
            for band in r.median_depth:
                assert band in r.cadence_depths[window]

    def test_two_band_strategy_shares_multi_exposure_depth(self, tmp_path):
        """Under a 1:1 strip cycle each band gets half the exposures in a
        multi-exposure window, so its coadd depth is 1.25 log10(2) shallower
        than a single-band strategy; single exposures are untouched.
        """
        depths = {}
        for label, options in (("single", ["g"]), ("alternating", ["g", "rho", "g", "rho"])):
            cfg = Config(sim_version="test")
            cfg.telescope.psf_model = "gaussian"  # deterministic PSF; the delivered default draws pixel phases
            cfg.filter_strategy.options = options
            mc = ForwardMC(
                cfg,
                sample_priors=[],
                output_dir=str(tmp_path / label),
                cadence_windows=("1sec", "1night", "1week"),
                lunar_mode="dark",
            )
            depths[label] = mc.run(n_samples=1, seed=3)[0].cadence_depths
        expected = 1.25 * np.log10(2.0)
        assert depths["alternating"]["1sec"]["g"] == pytest.approx(depths["single"]["1sec"]["g"], abs=1e-6)
        for window in ("1night", "1week"):
            loss = depths["single"][window]["g"] - depths["alternating"][window]["g"]
            assert loss == pytest.approx(expected, abs=0.03), f"{window}: {loss:.3f} vs {expected:.3f}"

    @pytest.mark.parametrize("seed", [7, 42, 123, 456, 999])
    def test_longer_cadence_is_deeper(self, tmp_path, seed):
        """Coadding more exposures must produce a fainter limiting magnitude.

        Parametrized over multiple seeds that each perturb seeing,
        transparency, and readnoise to verify the monotonicity property
        holds across a range of valid instrument configurations.
        """
        rng = np.random.default_rng(seed)
        cfg = Config(sim_version="test")
        cfg.observatory.seeing_mean = float(rng.uniform(0.6, 1.5))
        cfg.observatory.transparency = float(rng.uniform(0.55, 0.95))
        cfg.telescope.readnoise = float(rng.uniform(0.8, 2.5))

        mc = ForwardMC(
            cfg,
            sample_priors=[],
            output_dir=str(tmp_path / f"seed_{seed}"),
            cadence_windows=("1min", "1hour", "1night"),
        )
        results = mc.run(n_samples=1, seed=seed)
        r = results[0]

        for band in r.median_depth:
            m_1min = r.cadence_depths["1min"][band]
            m_1hr = r.cadence_depths["1hour"][band]
            m_1n = r.cadence_depths["1night"][band]
            assert m_1hr > m_1min, (
                f"1hour ({m_1hr:.2f}) should be deeper than 1min ({m_1min:.2f}) "
                f"for {band} (seeing={cfg.observatory.seeing_mean:.2f}, "
                f"transparency={cfg.observatory.transparency:.2f}, "
                f"readnoise={cfg.telescope.readnoise:.2f})"
            )
            assert m_1n > m_1hr, (
                f"1night ({m_1n:.2f}) should be deeper than 1hour ({m_1hr:.2f}) "
                f"for {band} (seeing={cfg.observatory.seeing_mean:.2f}, "
                f"transparency={cfg.observatory.transparency:.2f}, "
                f"readnoise={cfg.telescope.readnoise:.2f})"
            )

    def test_single_exposure_cadence_matches_median_depth(self, tmp_path):
        """The '1min' cadence window (coadd_n=1) should match single-exposure depth."""
        cfg = Config(sim_version="test")
        mc = ForwardMC(
            cfg,
            sample_priors=[],
            output_dir=str(tmp_path),
            cadence_windows=("1min",),
        )
        results = mc.run(n_samples=1, seed=42)
        r = results[0]

        for band in r.median_depth:
            assert r.cadence_depths["1min"][band] == pytest.approx(r.median_depth[band], abs=1e-6)

    def test_cadence_depths_in_dataframe(self, tmp_path):
        cfg = Config(sim_version="test")
        mc = ForwardMC(
            cfg,
            sample_priors=[],
            output_dir=str(tmp_path),
            cadence_windows=("1min", "1night"),
        )
        mc.run(n_samples=2, seed=42)
        df = mc.to_dataframe()

        bands = list(mc.results[0].median_depth.keys())
        for band in bands:
            assert f"cadence_1min_{band}" in df.columns
            assert f"cadence_1night_{band}" in df.columns

    def test_provenance_written_next_to_results(self, tmp_path):
        import json

        cfg = Config(sim_version="test")
        mc = ForwardMC(cfg, sample_priors=[], output_dir=str(tmp_path), cadence_windows=("1night",))
        mc.run(n_samples=1, seed=1)
        prov = json.loads((tmp_path / "provenance.json").read_text())
        assert prov["product"] == "forward_mc"
        assert prov["n_samples"] == 1
        assert len(prov["git_hash"]) == 40
        assert isinstance(prov["git_dirty"], bool)

    def test_cadence_depths_in_saved_json(self, tmp_path):
        import json

        cfg = Config(sim_version="test")
        mc = ForwardMC(
            cfg,
            sample_priors=[],
            output_dir=str(tmp_path),
            cadence_windows=("1min", "1night"),
        )
        mc.run(n_samples=1, seed=42)

        with open(tmp_path / "mc_results.json") as f:
            data = json.load(f)

        assert "cadence_depths" in data[0]
        assert "1min" in data[0]["cadence_depths"]
        assert "1night" in data[0]["cadence_depths"]

    def test_evaluate_single_image_depth_returns_psf_fwhm(self, tmp_path):
        """Calling without cadence_n_eff returns (depths, psf_fwhm) 2-tuple."""
        from argus_sim.ab_system import ABPhot
        from argus_sim.observatory import Observatory
        from argus_sim.photon_budget import NoiseBudget
        from argus_sim.psf import GaussianPSF
        from argus_sim.throughput import SystemThroughput

        cfg = Config(sim_version="test")
        obs = Observatory(config=cfg)
        tp_loss = cfg.telescope.throughput_loss
        throughputs = SystemThroughput(throughput_loss=tp_loss)
        ab = ABPhot(
            collecting_area=cfg.telescope.collecting_area,
            throughput_loss=tp_loss,
            throughput=throughputs,
        )
        noise_budget = NoiseBudget(
            read_noise=cfg.telescope.readnoise * u.electron,
            dark_current=cfg.telescope.dark_current * u.electron / u.second,
        )
        psf_model = GaussianPSF(
            stamp_size=cfg.survey.stamp_size,
            stamp_resolution=cfg.survey.stamp_resolution,
            plate_scale=cfg.telescope.plate_scale,
            pixel_size=cfg.telescope.pixel_size,
            rms_spot_size=cfg.telescope.spot_size,
        )
        band_params = build_band_params(set(cfg.filter_strategy.options), throughputs, ab)
        grid = build_ratchet_grid()

        result = evaluate_single_image_depth(
            obs=obs,
            noise_budget=noise_budget,
            psf_model=psf_model,
            ab=ab,
            band_params=band_params,
            grid_point=grid[0],
            seeing=cfg.observatory.seeing_mean,
            transparency=cfg.observatory.transparency,
            pixel_phase=None,
            cfg=cfg,
        )
        assert isinstance(result, DepthResult)
        assert isinstance(result.single_depths, dict)
        assert isinstance(result.psf_fwhm, dict)
        assert isinstance(result.optical_psf_size, dict)
        assert isinstance(result.cadence_depths, dict)
        for band in result.single_depths:
            assert band in result.psf_fwhm
            assert result.psf_fwhm[band] > 0
            assert band in result.optical_psf_size
            assert result.optical_psf_size[band] > 0
            assert result.optical_psf_size[band] <= result.psf_fwhm[band]


class TestSaturationMagnitude:
    """Property tests for saturation magnitude in depth output."""

    def _setup_depth_eval(self, tmp_path):
        from argus_sim.ab_system import ABPhot
        from argus_sim.observatory import Observatory
        from argus_sim.photon_budget import NoiseBudget
        from argus_sim.psf import GaussianPSF
        from argus_sim.throughput import SystemThroughput

        cfg = Config(sim_version="test")
        obs = Observatory(config=cfg)
        tp_loss = cfg.telescope.throughput_loss
        throughputs = SystemThroughput(throughput_loss=tp_loss)
        ab = ABPhot(
            collecting_area=cfg.telescope.collecting_area,
            throughput_loss=tp_loss,
            throughput=throughputs,
        )
        noise_budget = NoiseBudget(
            read_noise=cfg.telescope.readnoise * u.electron,
            dark_current=cfg.telescope.dark_current * u.electron / u.second,
        )
        psf_model = GaussianPSF(
            stamp_size=cfg.survey.stamp_size,
            stamp_resolution=cfg.survey.stamp_resolution,
            plate_scale=cfg.telescope.plate_scale,
            pixel_size=cfg.telescope.pixel_size,
            rms_spot_size=cfg.telescope.spot_size,
        )
        band_params = build_band_params(set(cfg.filter_strategy.options), throughputs, ab)
        grid = build_ratchet_grid()
        return cfg, obs, noise_budget, psf_model, ab, band_params, grid

    def test_sat_mag_returned_per_band(self, tmp_path):
        """evaluate_single_image_depth returns sat_mag dict with same bands as depths."""
        cfg, obs, noise_budget, psf_model, ab, band_params, grid = self._setup_depth_eval(tmp_path)

        dr = evaluate_single_image_depth(
            obs=obs,
            noise_budget=noise_budget,
            psf_model=psf_model,
            ab=ab,
            band_params=band_params,
            grid_point=grid[0],
            seeing=cfg.observatory.seeing_mean,
            transparency=cfg.observatory.transparency,
            pixel_phase=None,
            cfg=cfg,
        )
        assert isinstance(dr.sat_mags, dict)
        assert set(dr.sat_mags.keys()) == set(dr.single_depths.keys())

    def test_sat_mag_brighter_than_limmag(self, tmp_path):
        """For any valid config, saturation magnitude is brighter (smaller) than limiting magnitude."""
        cfg, obs, noise_budget, psf_model, ab, band_params, grid = self._setup_depth_eval(tmp_path)

        for gp in grid[:9]:
            dr = evaluate_single_image_depth(
                obs=obs,
                noise_budget=noise_budget,
                psf_model=psf_model,
                ab=ab,
                band_params=band_params,
                grid_point=gp,
                seeing=cfg.observatory.seeing_mean,
                transparency=cfg.observatory.transparency,
                pixel_phase=None,
                cfg=cfg,
            )
            for band in dr.single_depths:
                assert dr.sat_mags[band] < dr.single_depths[band], (
                    f"sat_mag ({dr.sat_mags[band]:.2f}) should be brighter than "
                    f"limmag ({dr.single_depths[band]:.2f}) for {band}"
                )

    def test_sat_mag_in_mc_run(self, tmp_path):
        """ForwardMC results include median_sat_mag per band."""
        cfg = Config(sim_version="test")
        mc = ForwardMC(
            cfg,
            sample_priors=[],
            output_dir=str(tmp_path),
        )
        results = mc.run(n_samples=1, seed=42)
        r = results[0]

        assert hasattr(r, "median_sat_mag")
        assert len(r.median_sat_mag) > 0
        for band in r.median_depth:
            assert band in r.median_sat_mag
            assert r.median_sat_mag[band] < r.median_depth[band]

    def test_sat_mag_with_cadence(self, tmp_path):
        """Saturation magnitude is returned alongside cadence depths."""
        cfg, obs, noise_budget, psf_model, ab, band_params, grid = self._setup_depth_eval(tmp_path)

        cadence_n_eff = {"1min": 1.0, "1hour": 60.0}
        dr = evaluate_single_image_depth(
            obs=obs,
            noise_budget=noise_budget,
            psf_model=psf_model,
            ab=ab,
            band_params=band_params,
            grid_point=grid[0],
            seeing=cfg.observatory.seeing_mean,
            transparency=cfg.observatory.transparency,
            pixel_phase=None,
            cfg=cfg,
            cadence_n_eff=cadence_n_eff,
        )
        assert isinstance(dr.sat_mags, dict)
        for band in dr.single_depths:
            assert band in dr.sat_mags
            assert dr.sat_mags[band] < dr.single_depths[band]


class TestTrackingElongation:
    """Tests for tracking elongation integration in the forward MC."""

    def test_elongation_degrades_depth(self, tmp_path):
        """Nonzero tracking elongation produces shallower depth than zero."""
        cfg = Config(sim_version="test")

        mc_no_elong = ForwardMC(
            cfg,
            sample_priors=[],
            output_dir=str(tmp_path / "no_elong"),
            cadence_windows=("1min",),
            max_tracking_elongation=0.0,
        )
        mc_with_elong = ForwardMC(
            cfg,
            sample_priors=[],
            output_dir=str(tmp_path / "with_elong"),
            cadence_windows=("1min",),
            max_tracking_elongation=5.0,
        )

        results_no = mc_no_elong.run(n_samples=1, seed=42)
        results_with = mc_with_elong.run(n_samples=1, seed=42)

        for band in results_no[0].median_depth:
            assert results_with[0].median_depth[band] <= results_no[0].median_depth[band], (
                f"Elongation should degrade depth for {band}"
            )

    def test_elongation_recorded_in_theta(self, tmp_path):
        """Tracking elongation and angle are recorded in the theta dict."""
        cfg = Config(sim_version="test")
        mc = ForwardMC(
            cfg,
            sample_priors=[],
            output_dir=str(tmp_path),
            cadence_windows=("1min",),
            max_tracking_elongation=2.0,
        )
        results = mc.run(n_samples=1, seed=42)
        theta = results[0].theta

        has_elongation = any("tracking_elongation" in k for k in theta)
        has_angle = any("tracking_angle" in k for k in theta)
        assert has_elongation, "tracking_elongation not in theta"
        assert has_angle, "tracking_angle not in theta"

    def test_zero_max_elongation_draws_zero(self, tmp_path):
        """When max_tracking_elongation=0, all elongation draws are 0."""
        cfg = Config(sim_version="test")
        mc = ForwardMC(
            cfg,
            sample_priors=[],
            output_dir=str(tmp_path),
            cadence_windows=("1min",),
            max_tracking_elongation=0.0,
        )
        results = mc.run(n_samples=1, seed=42)
        theta = results[0].theta

        for k, v in theta.items():
            if "tracking_elongation" in k:
                assert v == 0.0


class TestSensitivityTable:
    def _make_mock_df(self, n=200, seed=42):
        """Build a synthetic MC output where depth depends on two parameters."""
        import pandas as pd

        rng = np.random.default_rng(seed)
        readnoise = rng.normal(1.4, 0.2, n)
        seeing = rng.lognormal(np.log(0.87), 0.15, n)
        solar = rng.uniform(0.8, 2.0, n)

        # Depth depends strongly on seeing, weakly on readnoise, not on solar
        depth = 21.0 - 1.5 * (seeing - 0.87) - 0.3 * (readnoise - 1.4) + rng.normal(0, 0.05, n)

        return pd.DataFrame(
            {
                "theta_telescope.readnoise": readnoise,
                "theta_observatory.seeing_mean": seeing,
                "theta_observatory.solar_activity": solar,
                "median_depth_g": depth,
            }
        )

    def test_returns_dataframe_with_expected_columns(self):
        df = self._make_mock_df()
        result = compute_sensitivity(df, depth_col="median_depth_g")
        assert "parameter" in result.columns
        assert "spearman_rho" in result.columns
        assert "p_value" in result.columns
        assert "depth_range_16_84" in result.columns

    def test_all_theta_params_present(self):
        df = self._make_mock_df()
        result = compute_sensitivity(df)
        params = set(result["parameter"])
        assert "telescope.readnoise" in params
        assert "observatory.seeing_mean" in params
        assert "observatory.solar_activity" in params

    def test_sorted_by_absolute_correlation(self):
        df = self._make_mock_df()
        result = compute_sensitivity(df)
        rhos = result["spearman_rho"].abs().values
        assert all(rhos[i] >= rhos[i + 1] for i in range(len(rhos) - 1))

    def test_seeing_dominates(self):
        df = self._make_mock_df()
        result = compute_sensitivity(df)
        # Seeing should have the strongest correlation
        assert result.iloc[0]["parameter"] == "observatory.seeing_mean"
        assert abs(result.iloc[0]["spearman_rho"]) > 0.5

    def test_solar_activity_is_weak(self):
        df = self._make_mock_df()
        result = compute_sensitivity(df)
        solar_row = result[result["parameter"] == "observatory.solar_activity"].iloc[0]
        assert abs(solar_row["spearman_rho"]) < 0.2

    def test_depth_range_sign_matches_correlation(self):
        df = self._make_mock_df()
        result = compute_sensitivity(df)
        seeing_row = result[result["parameter"] == "observatory.seeing_mean"].iloc[0]
        # Worse seeing → shallower depth, so high-seeing quartile has
        # lower depth → negative depth_range
        assert seeing_row["depth_range_16_84"] < 0

    def test_raises_on_missing_theta(self):
        import pandas as pd

        df = pd.DataFrame({"x": [1, 2, 3], "median_depth_g": [20, 21, 22]})
        with pytest.raises(ValueError, match="No theta"):
            compute_sensitivity(df)

    def test_raises_on_missing_depth(self):
        import pandas as pd

        df = pd.DataFrame({"theta_x": [1, 2, 3], "y": [20, 21, 22]})
        with pytest.raises(ValueError, match="No median_depth"):
            compute_sensitivity(df)


class TestPlotPriors:
    def test_returns_figure(self):
        import matplotlib

        matplotlib.use("Agg")
        fig = plot_priors(n_draws=500)
        assert fig is not None
        assert len(fig.axes) > 0
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_save_to_file(self, tmp_path):
        import matplotlib

        matplotlib.use("Agg")
        path = str(tmp_path / "priors.png")
        fig = plot_priors(n_draws=500, save_path=path)
        assert (tmp_path / "priors.png").exists()
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_custom_priors(self):
        import matplotlib

        matplotlib.use("Agg")
        custom_sample = [Prior("telescope.readnoise", "normal", {"mean": 1.4, "std": 0.2})]
        custom_ratchet = [
            Prior("observatory.seeing_mean", "lognormal", {"mean": np.log(0.87), "std": 0.15}),
            Prior("observatory.transparency", "normal", {"mean": 0.75, "std": 0.03}),
        ]
        fig = plot_priors(
            sample_priors=custom_sample,
            ratchet_priors=custom_ratchet,
            n_draws=200,
        )
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotSensitivity:
    def _make_mock_df(self, n=200, seed=42):
        import pandas as pd

        rng = np.random.default_rng(seed)
        readnoise = rng.normal(1.4, 0.2, n)
        seeing = rng.lognormal(np.log(0.87), 0.15, n)
        solar = rng.uniform(0.8, 2.0, n)
        depth = 21.0 - 1.5 * (seeing - 0.87) - 0.3 * (readnoise - 1.4) + rng.normal(0, 0.05, n)
        return pd.DataFrame(
            {
                "theta_telescope.readnoise": readnoise,
                "theta_observatory.seeing_mean": seeing,
                "theta_observatory.solar_activity": solar,
                "median_depth_g": depth,
            }
        )

    def test_from_mc_df(self):
        import matplotlib

        matplotlib.use("Agg")
        df = self._make_mock_df()
        fig = plot_sensitivity(mc_df=df)
        assert len(fig.axes) == 2
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_from_precomputed_table(self):
        import matplotlib

        matplotlib.use("Agg")
        df = self._make_mock_df()
        st = compute_sensitivity(df)
        fig = plot_sensitivity(sensitivity_df=st)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_save_to_file(self, tmp_path):
        import matplotlib

        matplotlib.use("Agg")
        df = self._make_mock_df()
        path = str(tmp_path / "sensitivity.png")
        fig = plot_sensitivity(mc_df=df, save_path=path)
        assert (tmp_path / "sensitivity.png").exists()
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_raises_without_input(self):
        with pytest.raises(ValueError, match="Provide either"):
            plot_sensitivity()


class TestSolarActivityPrior:
    """The solar-activity prior is the 2028-2033 monthly table."""

    def test_empirical_prior_draws_from_the_table(self):
        """Every draw is one of the tabulated monthly values and the mean matches the table."""
        from argus_sim.forward_mc import DEFAULT_SAMPLE_PRIORS, SOLAR_ACTIVITY_2028_2033

        prior = next(p for p in DEFAULT_SAMPLE_PRIORS if p.field == "observatory.solar_activity")
        assert prior.dist == "empirical"
        rng = np.random.default_rng(0)
        draws = np.array([prior.draw(rng) for _ in range(3000)])
        assert set(np.round(draws, 4)) <= set(np.round(SOLAR_ACTIVITY_2028_2033, 4))
        assert draws.mean() == pytest.approx(np.mean(SOLAR_ACTIVITY_2028_2033), abs=0.02)
        assert 0.8 <= draws.min() and draws.max() <= 2.0
        assert len(SOLAR_ACTIVITY_2028_2033) == 72


class TestBlockDuty:
    """Strict alternation of 15-exposure blocks (one 16-minute pointing each), the default duty model.

    Windows start on a block boundary; the starting block is uniform over the
    cycle.  A window of at most one block is undivided in the on-duty band with
    f_on = cycle share; 30 min at 1:1 is 15 + 15 with f_on = 1; longer windows
    carry the exact cycle share in the phase mean with f_on = 1.
    """

    ONE_ONE = ["g", "rho", "g", "rho"]
    TWO_ONE = ["g", "g", "rho"]

    def test_window_within_one_block_is_undivided(self):
        """T <= 15 min: every exposure is in the on-duty band; f_on is the cycle share."""
        from argus_sim.forward_mc import block_duty

        for n_all in (1.0, 5.0, 15.0):
            d = block_duty(self.ONE_ONE, n_all, 60.0)
            for band in ("g", "rho"):
                assert d[band]["f_on"] == pytest.approx(0.5)
                assert d[band]["n_on_min"] == d[band]["n_on_mean"] == d[band]["n_on_max"] == pytest.approx(n_all)
        d = block_duty(self.TWO_ONE, 15.0, 60.0)
        assert d["g"]["f_on"] == pytest.approx(2 / 3) and d["g"]["n_on_mean"] == pytest.approx(15.0)
        assert d["rho"]["f_on"] == pytest.approx(1 / 3) and d["rho"]["n_on_mean"] == pytest.approx(15.0)

    def test_thirty_minutes(self):
        """30 min at 1:1 is 15 + 15 with f_on 1; at 2:1 it is 30 g / 15 g + 15 rho / 15 rho + 15 g."""
        from argus_sim.forward_mc import block_duty

        d = block_duty(self.ONE_ONE, 30.0, 60.0)
        for band in ("g", "rho"):
            assert d[band]["f_on"] == 1.0
            assert d[band]["n_on_min"] == d[band]["n_on_mean"] == d[band]["n_on_max"] == pytest.approx(15.0)
        d = block_duty(self.TWO_ONE, 30.0, 60.0)
        assert d["g"]["f_on"] == 1.0
        assert (d["g"]["n_on_min"], d["g"]["n_on_mean"], d["g"]["n_on_max"]) == pytest.approx((15.0, 20.0, 30.0))
        assert d["rho"]["f_on"] == pytest.approx(2 / 3)
        assert (d["rho"]["n_on_min"], d["rho"]["n_on_mean"], d["rho"]["n_on_max"]) == pytest.approx((15.0, 15.0, 15.0))

    def test_one_hour_at_two_to_one(self):
        """Four blocks at 2:1: g sees 45, 45, 30 (mean 40 = 2/3 of 60); rho 15, 15, 30 (mean 20)."""
        from argus_sim.forward_mc import block_duty

        d = block_duty(self.TWO_ONE, 60.0, 60.0)
        assert d["g"]["f_on"] == 1.0 and d["rho"]["f_on"] == 1.0
        assert (d["g"]["n_on_min"], d["g"]["n_on_mean"], d["g"]["n_on_max"]) == pytest.approx((30.0, 40.0, 45.0))
        assert (d["rho"]["n_on_min"], d["rho"]["n_on_mean"], d["rho"]["n_on_max"]) == pytest.approx((15.0, 20.0, 30.0))

    @pytest.mark.parametrize("n_all", [60.0, 510.0, 660.0, 2345.7, 1e5 + 0.3])
    def test_long_windows_carry_the_exact_cycle_share_in_the_mean(self, n_all):
        """Beyond one cycle every phase is on duty and the phase mean is n_all times the cycle share."""
        from argus_sim.forward_mc import band_shares, block_duty

        for cycle in (self.ONE_ONE, self.TWO_ONE):
            shares = band_shares(cycle)
            d = block_duty(cycle, n_all, 60.0)
            for band, share in shares.items():
                assert d[band]["f_on"] == 1.0
                assert d[band]["n_on_mean"] == pytest.approx(n_all * share, rel=1e-9)
                assert d[band]["n_on_min"] <= d[band]["n_on_mean"] <= d[band]["n_on_max"]
                assert d[band]["n_on_max"] - d[band]["n_on_min"] <= 15.0 * (len(cycle) - 1) + 1e-9

    def test_bright_one_second_exposures(self):
        """At 1 s exposures a block is 900 exposures: a 1-min window is undivided, an hour is split."""
        from argus_sim.forward_mc import block_duty

        d = block_duty(self.ONE_ONE, 60.0, 1.0)
        assert d["g"]["f_on"] == 0.5 and d["g"]["n_on_mean"] == pytest.approx(60.0)
        d = block_duty(self.ONE_ONE, 3600.0, 1.0)
        assert d["g"]["f_on"] == 1.0 and d["g"]["n_on_mean"] == pytest.approx(1800.0)

    def test_spec_uses_block_model(self):
        """_cadence_n_eff_spec: block gives the phase counts; single band gives a float."""
        from argus_sim.forward_mc import _cadence_n_eff_spec

        spec = _cadence_n_eff_spec("15min", "winter", 1.0, 60.0, 0.5, "g", "block", self.ONE_ONE)
        assert spec["share_mean"] == pytest.approx(7.5)
        assert spec["on_duty_p50"] == pytest.approx(15.0) and spec["f_on"] == 0.5
        spec = _cadence_n_eff_spec("1night", "winter", 1.0, 60.0, 0.5, "g", "block", self.ONE_ONE)
        assert spec["on_duty_p50"] == pytest.approx(spec["share_mean"]) and spec["f_on"] == 1.0
        assert _cadence_n_eff_spec("1night", "winter", 1.0, 60.0, 1.0, "g", "block", ["g"]) == pytest.approx(660.0)
        with pytest.raises(ValueError):
            _cadence_n_eff_spec("15min", "winter", 1.0, 60.0, 0.5, "g", "cells", self.ONE_ONE)

    def test_forward_mc_block_depths(self, tmp_path):
        """End to end: on-duty 15-min depth is +1.25 log10(2) over the share mean; a night matches it; provenance says block."""
        import json

        cfg = Config(sim_version="test")
        cfg.filter_strategy.options = ["g", "rho", "g", "rho"]
        mc = ForwardMC(
            cfg,
            sample_priors=[],
            output_dir=str(tmp_path),
            cadence_windows=("1sec", "15min", "30min", "1night"),
            lunar_mode="dark",
        )
        r = mc.run(n_samples=1, seed=3)[0]
        assert set(r.cadence_depths_on_duty) == {"1sec", "15min", "30min", "1night"}
        for b in ("g", "rho"):
            on = r.cadence_depths_on_duty
            assert on["1sec"][b]["p50"] == pytest.approx(r.cadence_depths["1sec"][b]) and on["1sec"][b]["f_on"] == 0.5
            assert on["15min"][b]["p50"] - r.cadence_depths["15min"][b] == pytest.approx(1.25 * np.log10(2.0), abs=0.03)
            assert on["15min"][b]["f_on"] == 0.5
            assert on["30min"][b]["p50"] == pytest.approx(r.cadence_depths["30min"][b], abs=1e-6)
            assert on["30min"][b]["f_on"] == 1.0
            assert on["1night"][b]["p50"] == pytest.approx(r.cadence_depths["1night"][b], abs=1e-6)
            assert on["1night"][b]["p16"] <= on["1night"][b]["p50"] <= on["1night"][b]["p84"]
        prov = json.load(open(tmp_path / "provenance.json"))
        assert prov["duty_model"] == "block" and prov["filter_cycle"] == ["g", "rho", "g", "rho"]
        with pytest.raises(ValueError):
            ForwardMC(cfg, sample_priors=[], output_dir=str(tmp_path), duty_model="cells")
