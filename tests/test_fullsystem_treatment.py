"""Full-system treatment in Survey and ForwardMC: exact signal efficiency, the delivered PSF by
default, PSF at each covering pair's field angle, seeing x X^0.6, the solar-activity prior and transparency draw in
Survey, and moon-masked rows left out of the sums."""

import astropy.table as tbl
import astropy.time as atime
import numpy as np
import pytest
import scipy.integrate as sing

from argus_sim import c
from argus_sim import survey as survey_mod
from argus_sim.config import Config
from argus_sim.forward_mc import SOLAR_ACTIVITY_2028_2033
from argus_sim.observatory import pickering_airmass, seeing_at_airmass
from argus_sim.psf import DeliveredPSF
from argus_sim.spectral_sky import build_band_params, signal_throughput


@pytest.fixture(scope="module")
def survey():
    return survey_mod.Survey(with_cradle=False)


@pytest.fixture(scope="module")
def bp(survey):
    return build_band_params(["g", "rho"], survey.throughputs, survey.ab)


@pytest.fixture(scope="module")
def small_tables(survey):
    grid = survey_mod.SEEING_TABLE_ARCSEC
    try:
        survey_mod.SEEING_TABLE_ARCSEC = np.geomspace(0.6, 2.4, 5)
        return survey_mod.build_sharpness_tables(survey.psf, ["g", "rho"])
    finally:
        survey_mod.SEEING_TABLE_ARCSEC = grid


# ------------------------------------------------------------------ exact signal efficiency
@pytest.mark.parametrize("band", ["g", "rho"])
@pytest.mark.parametrize("airmass", [1.0, 1.37, 2.0])
def test_signal_throughput_is_the_exact_photon_rate_integral(survey, bp, band, airmass):
    tp = survey.throughputs
    nu = tp.freq.value
    R = np.asarray(tp.filt_freq[band])
    qe = np.asarray(tp.qe_freq.value)
    eta = sing.trapezoid(R * tp.optics[band] * qe * np.asarray(tp.filt_freq.atmosphere) ** airmass / nu, nu)
    eta /= sing.trapezoid(R / nu, nu)
    band_qe = float(bp[band]["band_qe"].value)
    np.testing.assert_allclose(float(signal_throughput(bp[band], airmass)) * band_qe, eta, rtol=2e-4)


def test_band_summaries_derive_from_the_table(bp):
    # signal_tp_no_atm is eta(0) / band_qe and atm_tp_am1 is eta(1) / eta(0), both from the exact table
    tp_g = bp["g"]
    eta = float(signal_throughput(tp_g, 1.0))
    assert eta > 0 and tp_g["signal_tp_table"][0][0] == 0.0
    np.testing.assert_allclose(tp_g["atm_tp_am1"], eta / tp_g["signal_tp_no_atm"], rtol=1e-3)


# ------------------------------------------------------------------ delivered PSF by default
def test_default_psf_is_the_delivered_psf(survey):
    cfg = Config(sim_version="t")
    assert cfg.telescope.psf_model == "delivered"
    assert cfg.telescope.psf_fits_path.endswith("delivered_psf.fits")
    assert cfg.telescope.ee50_profile_path is None
    assert isinstance(survey.psf, DeliveredPSF)
    assert survey.psf.ee50_profile is None
    assert len(survey.psf.field_angles("rho")) == 7


def test_fixed_field_angle_ignores_the_rng(survey):
    a = survey.psf.with_seeing(1.0, band="g", field_angle_deg=1.9, pixel_phase=(0, 0), rng=np.random.default_rng(1))
    b = survey.psf.with_seeing(1.0, band="g", field_angle_deg=1.9, pixel_phase=(0, 0), rng=np.random.default_rng(2))
    np.testing.assert_array_equal(a[0], b[0])
    on_axis = survey.psf.with_seeing(1.0, band="g", field_angle_deg=0.0, pixel_phase=(0, 0))[1]
    assert a[1] != on_axis  # the delivered EE50 grows with field angle


def test_sharpness_table_interpolates_the_direct_psf(survey, small_tables):
    angles, grid, S = small_tables["g"]
    assert S.shape == (7, 5) and np.all(np.diff(S, axis=1) < 0)  # sharper in better seeing
    fa, fwhm = float(angles[5]), 1.3  # between grid points 1.2 and 1.7
    direct = np.mean(
        [
            np.sum(survey.psf.with_seeing(fwhm, band="g", field_angle_deg=fa, pixel_phase=ph)[0] ** 2)
            for ph in survey_mod.PIXEL_PHASES
        ]
    )
    np.testing.assert_allclose(survey_mod.sharpness_at(small_tables["g"], fa, fwhm), direct, rtol=0.02)


# ------------------------------------------------------------------ seeing x X^0.6
def test_seeing_scales_as_airmass_to_three_fifths():
    np.testing.assert_allclose(seeing_at_airmass(1.0, np.array([1.0, 2.0])), [1.0, 2.0**0.6])


def test_forward_mc_evaluates_the_psf_at_the_grid_point_airmass(monkeypatch):
    import astropy.units as u

    from argus_sim import forward_mc as fm
    from argus_sim.ab_system import ABPhot
    from argus_sim.observatory import Observatory
    from argus_sim.photon_budget import NoiseBudget
    from argus_sim.psf import GaussianPSF
    from argus_sim.throughput import SystemThroughput

    cfg = Config(sim_version="t")
    tp = SystemThroughput(throughput_loss=1.0)
    ab = ABPhot(collecting_area=cfg.telescope.collecting_area, throughput_loss=1.0, throughput=tp)
    psf = GaussianPSF(
        stamp_size=cfg.survey.stamp_size,
        stamp_resolution=cfg.survey.stamp_resolution,
        plate_scale=cfg.telescope.plate_scale,
        pixel_size=cfg.telescope.pixel_size,
        rms_spot_size=cfg.telescope.spot_size,
    )
    seen = []
    orig = psf.with_seeing
    monkeypatch.setattr(psf, "with_seeing", lambda seeing, **kw: seen.append(seeing) or orig(seeing, **kw))
    grid = fm.build_ratchet_grid(lunar_mode="dark")
    pt = next(p for p in grid if p.airmass > 1.5)
    fm.evaluate_single_image_depth(
        Observatory(config=cfg),
        NoiseBudget(read_noise=1.2 * u.electron, dark_current=0.0 * u.electron / u.second),
        psf,
        ab,
        build_band_params(["g"], tp, ab),
        pt,
        0.9,
        1.0,
        None,
        cfg,
    )
    np.testing.assert_allclose(seen, [0.9 * pt.airmass**0.6])


# ------------------------------------------------------------------ solar activity and transparency in Survey
def test_solar_activity_prior_by_night():
    t = atime.Time(["2030-06-10T04:00", "2030-06-10T05:00", "2026-06-15T04:00", "2026-06-16T04:00"])
    nights = np.array([1, 1, 2, 3])
    s = survey_mod.solar_activity_by_ratchet(t, nights, seed=1)
    assert s[0] == s[1] == SOLAR_ACTIVITY_2028_2033[(2030 - 2028) * 12 + 5]
    assert s[2] in SOLAR_ACTIVITY_2028_2033 and s[3] in SOLAR_ACTIVITY_2028_2033
    np.testing.assert_array_equal(s, survey_mod.solar_activity_by_ratchet(t, nights, seed=1))


def test_transparency_prior_is_the_cloud_model_the_code_draws():
    import argus_sim.forward_mc as fm

    prior = fm.transparency_prior(fm.DEFAULT_RATCHET_PRIORS)
    assert prior.dist == "cloud_lognormal" and prior.params == fm.CLOUD_TRANSPARENCY
    assert fm.CLOUD_TRANSPARENCY == {
        "photometric_fraction": 0.78,
        "median_mag": 0.378,
        "sigma_ln": 1.143,
        "max_mag": 2.0,
    }
    rng = np.random.default_rng(3)
    tr = np.array([fm.draw_atmosphere(rng, 0.87, 0.15, prior)[1] for _ in range(20000)])
    extra = -2.5 * np.log10(tr)
    assert tr.max() == 1.0 and extra.max() <= 2.0 + 1e-12
    assert abs((tr == 1.0).mean() - 0.78) < 0.01  # photometric fraction (DES Y3 FGCM)
    cloudy = extra[tr < 1.0]
    assert abs(np.median(cloudy) - 0.378) < 0.02  # cloud-tail lognormal
    assert abs((extra > 0.7).mean() - 0.065) < 0.006 and abs((extra > 1.5).mean() - 0.025) < 0.004
    rng = np.random.default_rng(5)  # Prior.draw uses the same model
    assert abs(np.mean([prior.draw(rng) == 1.0 for _ in range(20000)]) - 0.78) < 0.01


def test_forward_mc_draws_the_transparency_prior(monkeypatch, tmp_path):
    import argus_sim.forward_mc as fm

    seen = []
    orig = fm.draw_atmosphere

    def spy(rng, sm, ss, prior, rho=0.5):
        seen.append(prior)
        return orig(rng, sm, ss, prior, rho)

    monkeypatch.setattr(fm, "draw_atmosphere", spy)
    cfg = Config(sim_version="t")
    cfg.telescope.psf_model = "gaussian"
    fm.ForwardMC(cfg, sample_priors=[], cadence_windows=("1min",), lunar_mode="dark", output_dir=str(tmp_path)).run(
        n_samples=1, seed=1
    )
    assert seen and all(p.dist == "cloud_lognormal" for p in seen)


def _grid(alt, az, band="g"):
    return tbl.Table(
        {"alt": np.asarray(alt, float), "az": np.asarray(az, float), "filter": np.array([band] * len(alt))}
    )


def test_worker_applies_transparency_solar_and_restores_config(survey, bp, small_tables, monkeypatch):
    import argus_sim.forward_mc as fm

    survey._sharpness_tables = small_tables
    t = atime.Time("2026-06-15T06:00:00")
    grid = _grid([80.0, 50.0], [10.0, 200.0])
    before = survey._config.observatory.solar_activity
    monkeypatch.setattr(fm, "draw_atmosphere", lambda rng, *a, **kw: (1.0, 1.0))
    clear = survey._proc_ratchet(0, t, bp, 0.87, 0.15, None, None, 1, n_dark=14, grid_tab=grid, solar_activity=0.8)
    monkeypatch.setattr(fm, "draw_atmosphere", lambda rng, *a, **kw: (1.0, 0.8))
    grey = survey._proc_ratchet(0, t, bp, 0.87, 0.15, None, None, 1, n_dark=14, grid_tab=grid, solar_activity=0.8)
    high = survey._proc_ratchet(0, t, bp, 0.87, 0.15, None, None, 1, n_dark=14, grid_tab=grid, solar_activity=2.0)
    assert survey._config.observatory.solar_activity == before
    np.testing.assert_allclose(grey["signal_tp"], 0.8 * clear["signal_tp"], rtol=1e-12)
    assert np.all(grey["bkg_e"] < clear["bkg_e"])  # the extraterrestrial sky is attenuated too
    assert np.all(high["bkg_e"] > grey["bkg_e"])  # airglow grows with solar activity
    np.testing.assert_allclose(clear["airmass"], pickering_airmass(grid["alt"].data))


def test_worker_marks_rows_near_the_moon(survey, bp, small_tables, monkeypatch):
    survey._sharpness_tables = small_tables
    t = atime.Time("2026-06-30T05:00:00")  # Moon up, near full
    m = survey.obs.sky_components_at(alt_deg=np.array([60.0]), az_deg=np.array([0.0]), t=t)
    import astropy.coordinates as crds
    import astropy.units as u

    loc = crds.EarthLocation(
        lat=c.observatory.latitude * u.deg, lon=c.observatory.longitude * u.deg, height=c.observatory.altitude * u.m
    )
    moon = crds.get_body("moon", t, loc).transform_to(crds.AltAz(obstime=t, location=loc))
    assert moon.alt.deg > 20 and m is not None
    grid = _grid([moon.alt.deg, 60.0], [moon.az.deg, (moon.az.deg + 180) % 360])
    res = survey._proc_ratchet(0, t, bp, 0.87, 0.15, None, None, 1, n_dark=14, grid_tab=grid, solar_activity=1.0)
    np.testing.assert_array_equal(res["masked"], [True, False])


# ------------------------------------------------------------------ per-pair weights and field angles
def test_per_pair_weights_equal_tile_weights_when_constant():
    from argus_sim.coverage_exact import Layout
    from argus_sim.tanseg_coverage import MinipixAccumulator, TansegCoverage, icrs_to_array_rotation

    # a 2-deg dec band through the footprint keeps the accumulators small (full sky is ~25 GB: probe, not test)
    eng = TansegCoverage(
        Layout.from_config("A170r_1200", ["g", "rho", "g", "rho"]), backend="numpy", dec_range=(30.0, 32.0)
    )
    R = icrs_to_array_rotation(atime.Time("2026-06-15T06:00"), 0.0)
    p = eng.pairs(R)
    w = np.linspace(1.0, 2.0, eng.n_tiles)
    fb, pb = eng.ota_band[p.full_ota], eng.ota_band[p.part_ota]
    a = MinipixAccumulator(eng, None)
    b = MinipixAccumulator(eng, None)
    a.add(p, {0: w, 1: w}, n_epochs=14)
    b.add(p, {k: (w[p.full_row[fb == k]], w[p.part_row[pb == k]]) for k in (0, 1)}, n_epochs=14)
    np.testing.assert_allclose(a.ivar, b.ivar, rtol=1e-6)
    assert np.array_equal(a.nobs, b.nobs)
    fa_full, fa_part = eng.pair_field_angles(p, R)
    assert len(p.full_row) > 0 and len(p.part_row) > 0  # the band does contain covering pairs
    assert len(fa_full) == len(p.full_row) and len(fa_part) == len(p.part_row)
    half_diag = np.degrees(np.arctan(np.hypot(eng.tl, eng.ts)))
    assert fa_full.min() >= 0 and fa_full.max() < half_diag  # a FULL pair's tile centre is inside the OTA
    assert fa_part.max() < half_diag  # a PARTIAL pair's covered centroid is inside too


# ------------------------------------------------------------------ delivered EE50 of the default PSF
SENSOR_DEG = (3.2529, 2.4425)
PHASES = ((0.25, 0.75), (0.75, 0.25), (0.0, 0.5), (0.5, 0.0))


@pytest.mark.parametrize("band", ["g", "rho"])
def test_delivered_psf_ee50_spans_two_to_three_pixels(band):
    # seeing-free pixelated EE50 of the default PSF: 2.0-3.0 px across the field, area mean ~2.3 px over the sensor
    z = DeliveredPSF.from_config(Config(sim_version="t"))
    ang = z.field_angles(band)
    px = (
        np.array(
            [
                np.mean([z.with_seeing(0.0, band=band, field_angle_deg=float(a), pixel_phase=ph)[1] for ph in PHASES])
                for a in ang
            ]
        )
        / z.plate_scale
    )
    assert px.min() == pytest.approx(2.0, abs=0.1) and px.max() == pytest.approx(3.0, abs=0.1)
    rng = np.random.default_rng(1)
    theta = np.hypot(
        rng.uniform(-SENSOR_DEG[0] / 2, SENSOR_DEG[0] / 2, 200000),
        rng.uniform(-SENSOR_DEG[1] / 2, SENSOR_DEG[1] / 2, 200000),
    )
    assert px[np.searchsorted((ang[1:] + ang[:-1]) / 2, theta)].mean() == pytest.approx(2.30, abs=0.05)


def test_ee50_profile_broadens_the_delivered_psf(tmp_path):
    # an optional profile with a larger target broadens the optics-only PSF to meet it; a smaller one does nothing
    cfg = Config(sim_version="t")
    base = DeliveredPSF.from_config(cfg)
    _, _, hfd0 = base.with_seeing(0.0, band="g", field_angle_deg=0.0, pixel_phase=(0, 0), return_optical_hfd=True)
    path = tmp_path / "profile.csv"
    path.write_text(f"band,theta_deg,target_um\ng,0.0,{2 * hfd0}\ng,2.1,{2 * hfd0}\nrho,0.0,1.0\nrho,2.1,1.0\n")
    cfg.telescope.ee50_profile_path = str(path)
    wide = DeliveredPSF.from_config(cfg)
    _, _, hfd1 = wide.with_seeing(0.0, band="g", field_angle_deg=0.0, pixel_phase=(0, 0), return_optical_hfd=True)
    assert hfd1 == pytest.approx(2 * hfd0, rel=0.03)
    same = [p.with_seeing(1.0, band="rho", field_angle_deg=1.0, pixel_phase=(0, 0))[0] for p in (base, wide)]
    np.testing.assert_array_equal(same[0], same[1])


# ------------------------------------------------------------------ weather and seasonal seeing on by default
def test_weather_and_seasonal_seeing_are_on_by_default():
    # the weather model gates nights; the DIMM seasons of observatory.site_seeing drive the seeing
    cfg = Config(sim_version="t")
    assert cfg.survey.apply_weather is True
    assert cfg.observatory.seasonal_seeing is True
    assert c.survey.apply_weather is True  # argussim.toml agrees


def test_driver_can_still_observe_every_night(tmp_path):
    from argus_sim import c as cfg_global
    from argus_sim.survey_driver import build_parser, configure

    was = cfg_global.survey.apply_weather
    try:
        base = ["--start", "2026-06-14T18:00", "--n-nights", "1", "--outdir", str(tmp_path)]
        p = build_parser()
        assert configure(p.parse_args(base), str(tmp_path))["apply_weather"] is True  # config default
        assert configure(p.parse_args([*base, "--no-weather"]), str(tmp_path))["apply_weather"] is False
        assert configure(p.parse_args([*base, "--weather"]), str(tmp_path))["apply_weather"] is True
    finally:
        cfg_global.survey.apply_weather = was


def test_seeing_prior_states_the_seasonal_model_that_is_drawn():
    import argus_sim.forward_mc as fm
    from argus_sim.observatory import seasonal_seeing_params, site_seeing

    prior = next(p for p in fm.DEFAULT_RATCHET_PRIORS if p.field == "observatory.seeing_mean")
    assert prior.dist == "seasonal_lognormal"
    assert prior.params["winter"] == site_seeing[1]["median"] == 1.24
    assert prior.params["rest"] == site_seeing[6]["median"] == 0.93
    # the per-epoch draw the code actually makes: winter apart from the rest
    assert seasonal_seeing_params(atime.Time("2026-01-15T06:00"))[0] == 1.24
    assert seasonal_seeing_params(atime.Time("2026-06-15T06:00"))[0] == 0.93
    rng = np.random.default_rng(0)
    d = np.array([prior.draw(rng) for _ in range(20000)])
    assert abs(np.median(d) - 0.97) < 0.03  # annual marginal, 3 winter months in 12
