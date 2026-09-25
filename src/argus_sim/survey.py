"""Schedule-driven survey simulation: the Survey class and its per-ratchet photometry workers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional
import os
import astroplan
import astropy.time as atime
import astropy.units as u
import astropy_healpix as ahpx
import multiprocessing as mp
import numpy as np
import numpy.typing as npt
import pandas as pd

from . import ABPhot, CradlePointing, GaussianPSF, NoiseBudget, SystemThroughput, DeliveredPSF, c, get_logger
from .cadence_accumulator import CadenceAccumulator, build_window_registry
from .observatory import (
    Observatory,
    observable_weather,
    pickering_airmass,
    seasonal_seeing_params,
    seeing_at_airmass,
)
from .spectral_sky import build_band_params, signal_throughput

if TYPE_CHECKING:
    from .config import Config


def sample_ratchets(
    times: atime.Time,
    nights: npt.ArrayLike,
    fraction: float,
    rng: np.random.Generator,
) -> tuple[atime.Time, npt.ArrayLike]:
    """Downsample ratchets while preserving chronological order."""
    n = int(np.round(len(times) * fraction))
    idx = rng.choice(len(times), n, replace=False)
    idx.sort()
    return times[idx], nights[idx]


def draw_seeing(seeing_mean: float, seeing_std: float, rng: np.random.Generator) -> float:
    """Draw atmospheric seeing from lognormal distribution."""
    return rng.lognormal(np.log(seeing_mean), seeing_std)


def draw_seasonal_seeing(epoch, rng: np.random.Generator) -> float:
    """Draw seeing from the seasonal site model (Barker et al. 2003)."""
    median, std = seasonal_seeing_params(epoch)
    return rng.lognormal(np.log(median), std)


# Sharpness sum(PSF^2) is tabulated per band over the PSF's field angles and a seeing grid, averaged over four
# pixel phases (a coadd's exposures land at random phases); rows look it up at their own field angle and at the
# seeing at their airmass instead of drawing one PSF per band per ratchet.
SEEING_TABLE_ARCSEC = np.geomspace(0.4, 4.0, 16)
PIXEL_PHASES = ((0.0, 0.0), (0.0, 0.5), (0.5, 0.0), (0.5, 0.5))
SEEING_TRANSPARENCY_RHO = 0.5  # ForwardMC's default seeing-transparency correlation


def build_sharpness_tables(psf, bands) -> dict:
    """Per band: (field angles [deg], seeing grid [arcsec], mean sharpness (n_angle, n_seeing))."""
    rng = np.random.default_rng(0)
    out = {}
    for band in bands:
        angles = psf.field_angles(band) if hasattr(psf, "field_angles") else np.array([0.0])
        phases = PIXEL_PHASES if isinstance(psf, DeliveredPSF) else ((0.0, 0.0),)
        S = np.empty((len(angles), len(SEEING_TABLE_ARCSEC)))
        for i, a in enumerate(angles):
            for j, fwhm in enumerate(SEEING_TABLE_ARCSEC):
                S[i, j] = np.mean(
                    [
                        np.sum(
                            psf.with_seeing(float(fwhm), band=band, field_angle_deg=float(a), pixel_phase=ph, rng=rng)[
                                0
                            ]
                            ** 2
                        )
                        for ph in phases
                    ]
                )
        out[band] = (np.asarray(angles, np.float64), SEEING_TABLE_ARCSEC.copy(), S)
    return out


def sharpness_at(table, field_angle_deg, seeing_arcsec) -> np.ndarray:
    """Sharpness at each (field angle, seeing).

    Uses the nearest tabulated angle (as DeliveredPSF) and interpolates log-linearly in seeing (clipped to the grid).
    """
    angles, grid, S = table
    fa = np.atleast_1d(np.asarray(field_angle_deg, np.float64))
    se = np.atleast_1d(np.asarray(seeing_arcsec, np.float64))
    fa, se = np.broadcast_arrays(fa, se)
    ia = np.searchsorted((angles[1:] + angles[:-1]) / 2, fa) if len(angles) > 1 else np.zeros(fa.shape, int)
    lg = np.log(grid)
    ls = np.log(np.clip(se, grid[0], grid[-1]))
    j = np.clip(np.searchsorted(lg, ls) - 1, 0, len(lg) - 2)
    f = (ls - lg[j]) / (lg[j + 1] - lg[j])
    lS = np.log(S)
    return np.exp((1 - f) * lS[ia, j] + f * lS[ia, j + 1])


def random_field_angle(psf, rng: np.random.Generator) -> float:
    """Return the field angle of a uniformly drawn detector position (the MOC engine's per-ratchet PSF position)."""
    nx = getattr(psf, "n_pixels_x", None)
    if nx is None:
        return 0.0
    x = rng.uniform(-nx / 2, nx / 2)
    y = rng.uniform(-psf.n_pixels_y / 2, psf.n_pixels_y / 2)
    return float(np.hypot(x, y) * psf.plate_scale / 3600.0)


def solar_activity_by_ratchet(times: atime.Time, nights, seed=None) -> np.ndarray:
    """Solar-activity index per ratchet from ForwardMC's 2028-2033 prior.

    The index is the month's value when the night falls in 2028-01..2033-12, otherwise one draw per night from
    the 72 monthly values.
    """
    from .forward_mc import SOLAR_ACTIVITY_2028_2033

    table = np.asarray(SOLAR_ACTIVITY_2028_2033, np.float64)
    rng = np.random.default_rng([0 if seed is None else int(seed), 2028])
    nights = np.asarray(nights)
    out = np.empty(len(nights))
    for n in np.unique(nights):
        sel = nights == n
        dt = times[np.nonzero(sel)[0][0]].datetime
        k = (dt.year - 2028) * 12 + dt.month - 1
        out[sel] = table[k] if 0 <= k < len(table) else rng.choice(table)
    return out


_worker_survey = None
_worker_cradle = None


def write_parquet_with_metadata(df, path: str, metadata: dict[str, str], compression: str = "lz4") -> None:
    """Write ``df`` to parquet with ``metadata`` merged into the schema metadata.

    pandas' ``to_parquet(custom_metadata=...)`` does not work with pyarrow 23,
    so the table is built and written through pyarrow directly.  Readers find
    the keys in ``pq.read_table(path).schema.metadata``.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df)
    meta = dict(table.schema.metadata or {})
    meta.update({str(k).encode(): str(v).encode() for k, v in metadata.items()})
    pq.write_table(table.replace_schema_metadata(meta), path, compression=compression)


def _init_worker(cradle, config=None, sharpness_tables=None):
    """Per-worker initializer: create one Survey and stash the cradle (and the parent's PSF sharpness tables)."""
    global _worker_survey, _worker_cradle
    _worker_survey = Survey(with_cradle=False, config=config)
    _worker_survey._sharpness_tables = sharpness_tables
    _worker_cradle = cradle


def _process_ratchet_mp(args):
    """Standalone function for multiprocessing."""
    (ratchet_num, t, band_params, seeing_mean, seeing_std, hpx_to_keep, rng_seed, n_dark, grid_tab, solar) = args
    result = _worker_survey._proc_ratchet(
        ratchet_num,
        t,
        band_params,
        seeing_mean,
        seeing_std,
        hpx_to_keep,
        _worker_cradle,
        rng_seed,
        n_dark=n_dark,
        grid_tab=grid_tab,
        solar_activity=solar,
    )
    return (ratchet_num, result)


class Survey:
    """Schedule-driven survey simulation.

    Takes a concrete observing schedule (list of ratchet start times) and
    computes limiting magnitude at every HEALPix pixel at every ratchet
    using real ephemeris for sky brightness components.  Instrument
    parameters are fixed from :class:`~argus_sim.config.Config`.

    The output is a time-resolved depth map: one depth per HEALPix pixel
    per ratchet over the full survey duration.  Use this to answer "given
    this specific observing schedule and pointing strategy, what does the
    survey achieve?"

    For uncertainty propagation over instrument parameters, see
    :class:`~argus_sim.forward_mc.ForwardMC`, which varies the instrument
    model across draws rather than simulating the full schedule.

    """

    def __init__(
        self,
        start: Optional[atime.Time] = None,
        end: Optional[atime.Time] = None,
        with_cradle: Optional[bool] = True,
        config: Config | None = None,
    ) -> None:
        """Initialize the Survey object.

        Parameters
        ----------
        start : astropy.time.Time, optional
            Survey start time. Defaults to config value.
        end : astropy.time.Time, optional
            Survey end time. Defaults to config value.
        with_cradle : bool, optional
            Whether to build the CradlePointing grid. Default True.
        config : Config, optional
            Configuration object. Falls back to the global ``c`` singleton
            when not provided.

        """
        self.log = get_logger(__name__)
        cfg = config if config is not None else c
        self._config = cfg

        if start is None:
            self.start = atime.Time(cfg.survey.start_date, scale="utc")
        if end is None:
            self.end = atime.Time(cfg.survey.end_date, scale="utc")

        self.collecting_area = cfg.telescope.collecting_area

        self.hpx = ahpx.HEALPix(order="nested", nside=2**cfg.survey.sim_depth, frame="cirs")
        self.basemap = np.zeros(self.hpx.npix)

        if with_cradle and cfg.survey.coverage == "moc":
            self.cradle = CradlePointing(contract_lon=True)
        else:
            self.cradle = None
        self.obs = Observatory(config=cfg)
        self.noise_budget = NoiseBudget(
            read_noise=cfg.telescope.readnoise * u.electron,
            dark_current=cfg.telescope.dark_current * u.electron / u.second,
        )
        all_filters = {band: True for band in cfg.filter_strategy.options}
        tp_loss = cfg.telescope.throughput_loss
        self.throughputs = SystemThroughput(throughput_loss=tp_loss, filters=all_filters)
        self.ab = ABPhot(collecting_area=self.collecting_area, throughput_loss=tp_loss, throughput=self.throughputs)

        if cfg.telescope.psf_model == "delivered":
            self.psf = DeliveredPSF.from_config(cfg)
        else:
            self.psf = GaussianPSF(
                stamp_size=cfg.survey.stamp_size,
                stamp_resolution=cfg.survey.stamp_resolution,
                plate_scale=cfg.telescope.plate_scale,
                pixel_size=cfg.telescope.pixel_size,
                rms_spot_size=cfg.telescope.spot_size,
            )
        self._sharpness_tables = None

    def sharpness_tables(self) -> dict:
        """PSF sharpness tables for every configured band (built once, a few seconds for the delivered PSF)."""
        if self._sharpness_tables is None:
            bands = sorted(set(self._config.filter_strategy.options))
            self._sharpness_tables = build_sharpness_tables(self.psf, bands)
        return self._sharpness_tables

    def set_cradle(self, cradle: CradlePointing) -> None:
        """Inject a pre-built CradlePointing, avoiding reconstruction cost."""
        self.cradle = cradle

    def _proc_ratchet(
        self,
        ratchet_num: int,
        t: atime.Time,
        band_params: dict,
        seeing_mean: float | None,
        seeing_std: float | None,
        hpx_to_keep: npt.ArrayLike,
        cradle: CradlePointing,
        rng_seed: int = None,
        n_dark: int | None = None,
        grid_tab=None,
        solar_activity: float | None = None,
    ):
        cfg = self._config
        if solar_activity is None:
            return self._proc_ratchet_inner(
                ratchet_num, t, band_params, seeing_mean, seeing_std, hpx_to_keep, cradle, rng_seed, n_dark, grid_tab
            )
        saved = cfg.observatory.solar_activity
        cfg.observatory.solar_activity = float(solar_activity)
        try:
            return self._proc_ratchet_inner(
                ratchet_num, t, band_params, seeing_mean, seeing_std, hpx_to_keep, cradle, rng_seed, n_dark, grid_tab
            )
        finally:
            cfg.observatory.solar_activity = saved

    def _proc_ratchet_inner(
        self, ratchet_num, t, band_params, seeing_mean, seeing_std, hpx_to_keep, cradle, rng_seed, n_dark, grid_tab
    ):
        from .forward_mc import DEFAULT_RATCHET_PRIORS, draw_atmosphere, transparency_prior

        log = get_logger(__name__)
        cfg = self._config
        rng = np.random.default_rng(rng_seed)

        log.debug(f"Ratchet {ratchet_num} start: {t.isot}")
        # Zenith seeing and grey transparency drawn as ForwardMC draws them (draw_atmosphere): the transparency is
        # the cloud/cirrus model of the prior table, per ratchet; clear nights themselves are the
        # weather model's (survey.apply_weather).
        if seeing_mean is None:
            sm, ss = seasonal_seeing_params(t)
        else:
            sm, ss = seeing_mean, seeing_std
        seeing, transparency = draw_atmosphere(
            rng, sm, ss, transparency_prior(DEFAULT_RATCHET_PRIORS), rho=SEEING_TRANSPARENCY_RHO
        )
        log.debug(f"Ratchet {ratchet_num} seeing: {seeing} asec, transparency {transparency}")

        # grid_tab given: the exact engine's covered tile centres (alt/az, filter), one row per (tile, band)
        cells = grid_tab is None
        if cells:
            grid_tab = cradle.to_hpx_table_at(t)
        if cells and cfg.survey.n_healpix is not None:
            grid_tab = grid_tab[np.isin(grid_tab["healpix"], hpx_to_keep)]

        sky_components = self.obs.sky_components_at(
            alt_deg=grid_tab["alt"].data,
            az_deg=grid_tab["az"].data,
            t=t,
        )
        for key in ("artif_s10",):
            if np.ndim(sky_components[key]) == 0:
                sky_components[key] = np.full(len(grid_tab), sky_components[key])
        sky_brightness = sky_components["v_mag"]
        moon_separation = np.broadcast_to(np.asarray(sky_components["moon_distance_deg"], float), (len(grid_tab),))
        moon_alt = sky_components["moon_alt"]
        sun_alt = sky_components["sun_alt"]

        illum_frac = astroplan.moon_illumination(t)
        if illum_frac > cfg.survey.highspeed_start_moonfrac:
            if cfg.survey.skip_fast_cadence:
                log.debug(f"Ratchet {ratchet_num} is bright time, skipping.")
                return None

            log.debug(f"Ratchet {ratchet_num} is bright time, exptime is {cfg.survey.fast_cadence_s} s.")
            exptime = cfg.survey.fast_cadence_s * u.second
        else:
            log.debug(f"Ratchet {ratchet_num} is dark time, exptime is {cfg.survey.base_cadence_s} s.")
            exptime = cfg.survey.base_cadence_s * u.second

        if n_dark is not None:  # dark exposures of the base cadence in this ratchet (coverage_exact.Schedule)
            n_exposures_per_ratchet = int(round(n_dark * cfg.survey.base_cadence_s / exptime.value))
        else:
            n_exposures_per_ratchet = int(cfg.survey.ratchet_len / (exptime.value / 60))
        log.debug(
            f"Assuming {n_exposures_per_ratchet} exposures per average ratchet (max {int(cfg.survey.ratchet_len / (exptime.value / 60))})."
        )
        epochs = [t + i * exptime for i in range(n_exposures_per_ratchet)]

        if sun_alt > cfg.survey.min_sun_alt:
            log.debug(f"Sun too high: {sun_alt}, skipping ratchet.")
            return None
        # rows within survey.min_moon_sep of the Moon are not observed: they are left out of the sums
        masked = moon_separation < cfg.survey.min_moon_sep
        log.debug(f"Masked {masked.sum()} rows due to moon proximity.")

        has_filter_col = "filter" in grid_tab.colnames
        if has_filter_col:
            pixel_bands = np.array(grid_tab["filter"])
        else:
            fallback = list(band_params.keys())[0]
            pixel_bands = np.full(len(grid_tab), fallback)

        tables = self.sharpness_tables()
        n = len(grid_tab)
        limmag = np.zeros(n)
        signal_tp = np.zeros(n)  # full-system signal throughput per row: exact eta(X) / band_qe x transparency
        bkg_e = np.zeros(n)  # per-pixel background variance of one exposure: read^2 + dark + sky (e-^2)
        airmass_all = np.ones(n)
        field_angle = np.full(n, np.nan)
        budget_arrays = {}

        for band in np.unique(pixel_bands):
            if band not in band_params:
                continue
            bp = band_params[band]
            sel = pixel_bands == band
            if not np.any(sel):
                continue

            background_e = bp["sky"].sky_electrons(
                airglow_s10=sky_components["airglow_s10"][sel],
                zodiacal_s10=sky_components["zodiacal_s10"][sel],
                starlight_s10=sky_components["starlight_s10"][sel],
                moon_s10=sky_components["moon_s10"][sel],
                artif_s10=sky_components["artif_s10"][sel],
                transparency=transparency,
            )
            background_e *= self.psf.plate_scale**2
            background_e = background_e * u.electron / u.second

            # Per-row airmass (Pickering 2002), full-system throughput and seeing at that airmass (X^0.6)
            alt_deg = grid_tab["alt"].data[sel]
            airmass = pickering_airmass(alt_deg)
            airmass_all[sel] = airmass
            effective_tp = signal_throughput(bp, airmass) * transparency
            signal_tp[sel] = effective_tp
            # One field position per band per ratchet for these rows (the MOC engine's model); the exact engine
            # re-evaluates the sharpness per covering pair at the pair's own field angle from bkg_e.
            fa = random_field_angle(self.psf, rng)
            field_angle[sel] = fa
            sharp = sharpness_at(tables[band], fa, seeing_at_airmass(seeing, airmass))
            bkg_e[sel] = (
                cfg.telescope.readnoise**2
                + cfg.telescope.dark_current * exptime.value
                + background_e.value * exptime.value
            )

            signal_e, budget = self.noise_budget.get_flux_at_snr(
                snr=cfg.survey.detection_snr,
                psf=None,
                eskyflux=background_e,
                exptime=exptime,
                signal_throughput=effective_tp,
                coadd_n=1,
                sharpness=sharp,
            )

            signal_e /= exptime
            signal_photons = signal_e / bp["band_qe"]
            limmag[sel] = self.ab.mag_from_photons(band, signal_photons)

            for key in budget.keys():
                if key not in budget_arrays:
                    budget_arrays[key] = np.zeros(n)
                budget_arrays[key][sel] = budget[key]

        if cfg.output.save_grid_tables and cells:
            distance = 10 ** ((limmag - cfg.survey.reference_magnitude + 5) / 5) * u.parsec
            volume = (4 / 3) * np.pi * distance**3 * cradle.moc.sky_fraction
            volume = volume.to(u.megaparsec**3).value

            base_tab = grid_tab.copy()
            base_tab["masked"] = masked
            base_tab["exptime"] = [exptime] * len(base_tab)
            base_tab["ratchnum"] = [ratchet_num] * len(base_tab)
            base_tab["moon_alt"] = [moon_alt] * len(base_tab)
            base_tab["moon_frac"] = [illum_frac] * len(base_tab)
            base_tab["limmag"] = limmag
            base_tab["sky_brightness"] = sky_brightness
            base_tab["band"] = pixel_bands
            base_tab["nested"] = [1] * len(volume)
            base_tab["nside"] = [self.hpx.nside] * len(volume)
            base_tab["signal_tp"] = signal_tp
            base_tab["seeing_zenith"] = [seeing] * len(base_tab)
            base_tab["transparency"] = [transparency] * len(base_tab)
            base_tab["field_angle"] = field_angle
            for key, arr in budget_arrays.items():
                base_tab[key] = arr

            if not cfg.output.metadata:
                base_tab.remove_columns(
                    [
                        "sky_brightness",
                        "above_atm_electrons",
                        "readnoise",
                        "dark_electrons",
                        "sky_electrons",
                        "sharpness",
                    ]
                )

            base_df = base_tab.to_pandas()
            date_str = t.datetime.strftime("%Y%m%d")
            outdir = f"{cfg.output.output_dir}/ratchets/{date_str}"
            os.makedirs(outdir, exist_ok=True)
            write_parquet_with_metadata(
                base_df,
                f"{outdir}/ratchet_{ratchet_num:07d}.parquet",
                {
                    "epoch_start_mjd": str(epochs[0].mjd),
                    "epoch_exptime_s": str(exptime.to_value(u.second)),
                    "n_epochs": str(len(epochs)),
                },
            )

        out = {
            "healpix": np.asarray(grid_tab["healpix"]) if cells else None,
            "noise": budget_arrays["noise"],
            "source_shot": budget_arrays["source_shot"],
            "signal_tp": signal_tp,
            "bkg_e": bkg_e,
            "airmass": airmass_all,
            "masked": masked,
            "exptime_s": float(exptime.to_value(u.second)),
            "band": pixel_bands,
            "n_epochs": len(epochs),
            "moon_frac": float(illum_frac),
            "seeing": float(seeing),
            "transparency": float(transparency),
            "solar_activity": float(cfg.observatory.solar_activity),
        }
        if cells:  # the MOC engine sums whatever rows come back: drop the moon-masked ones here
            keep = ~masked
            for k in ("healpix", "noise", "source_shot", "signal_tp", "bkg_e", "airmass", "masked", "band"):
                out[k] = np.asarray(out[k])[keep]
        return out

    def _run_exact(
        self,
        t_first,
        t_mid,
        theta_first,
        n_dark,
        nights,
        seeds,
        band_params,
        seeing_mean,
        seeing_std,
        band_zeropoints,
        solar=None,
    ):
        """Exact coverage on the HDPS tanseg/minipix grid (tanseg_coverage) with the CPU photometry chain.

        Single-process GPU path: the CPU pool is forked before CUDA is initialised and only computes
        photometry at covered tile centres; the main process owns the device, the geometry and the sums.
        """
        import json
        import os

        import astropy.table as tbl

        from .coverage_exact import Layout
        from .forward_mc import DEFAULT_RATCHET_PRIORS, transparency_prior
        from .skymap_shim import PANOPTES_COMMIT
        from .tanseg_coverage import (
            MinipixAccumulator,
            RatchetPairs,
            TansegCoverage,
            icrs_to_array_rotation,
            owned_mask,
            write_epoch_record,
        )

        cfg = self._config
        outdir = cfg.output.output_dir
        os.makedirs(os.path.join(outdir, "epochs"), exist_ok=True)
        if solar is None:
            solar = np.full(len(t_first), cfg.observatory.solar_activity)
        tables = self.sharpness_tables()  # built before the fork and handed to every worker
        pool = mp.Pool(cfg.nproc, initializer=_init_worker, initargs=(None, cfg, tables), maxtasksperchild=200)
        try:
            eng = TansegCoverage(Layout.from_config(), backend="auto")
            xp = eng.xp
            gpu = {"peak_used": 0, "total": None}

            def sample_gpu():
                if xp is not np:
                    free, total = xp.cuda.runtime.memGetInfo()
                    gpu["total"] = total
                    gpu["peak_used"] = max(gpu["peak_used"], total - free)

            if xp is not np:
                xp.get_default_memory_pool().set_limit(fraction=cfg.survey.gpu_pool_fraction)
                self.log.info(f"CuPy pool capped at {cfg.survey.gpu_pool_fraction:.2f} of device memory.")
            self.log.info(f"Exact coverage on {xp.__name__}: {eng.n_tiles} reachable tiles, bands {eng.bands}.")
            acc = MinipixAccumulator(eng, outdir)
            owned = owned_mask(eng.tanseg_id, xp)
            h = (lambda a: a.get()) if xp is not np else (lambda a: np.asarray(a))
            nights_out, n_done, n_total = [], 0, len(t_first)
            current_night = None
            night_moon, night_fast, night_atm = [], [], []

            def finish_night(night):
                import time as _time

                t0 = _time.time()
                summary = self._exact_night_summary(eng, acc, owned, band_zeropoints)
                t1 = _time.time()
                summary.update(acc.end_night(night, owned))
                if xp is not np:  # hand cached blocks back so later nights and library temporaries have room
                    xp.get_default_memory_pool().free_all_blocks()
                self.log.info(f"Night {night} summary {t1 - t0:.1f} s, flush {_time.time() - t1:.1f} s.")
                summary["mjd_min"] = float(t_first.mjd[nights == night].min())
                summary["mjd_max"] = float(t_first.mjd[nights == night].max())
                # from the scheduler's own per-ratchet decisions (astroplan illumination; fast cadence above
                # survey.highspeed_start_moonfrac)
                summary["n_ratchets"] = int(len(night_moon))
                if night_moon:
                    summary["moon_frac_min"] = float(min(night_moon))
                    summary["moon_frac_median"] = float(np.median(night_moon))
                    summary["moon_frac_max"] = float(max(night_moon))
                summary["n_fast_ratchets"] = int(sum(night_fast))
                if night_atm:
                    a = np.array(night_atm, float)
                    summary["seeing_zenith_median"] = float(np.median(a[:, 0]))
                    summary["transparency_min"] = float(a[:, 1].min())
                    summary["transparency_median"] = float(np.median(a[:, 1]))
                    summary["solar_activity"] = float(a[0, 2])
                    summary["moon_masked_pairs"] = int(a[:, 3].sum())
                night_moon.clear()
                night_fast.clear()
                night_atm.clear()
                sample_gpu()
                summary["gpu_device_used_peak_bytes"] = int(gpu["peak_used"])
                nights_out.append(summary)
                self.log.info(f"Night {night}: " + ", ".join(f"{k} {v}" for k, v in summary.items() if "median" in k))

            batch = max(2, 2 * cfg.nproc)
            for s0 in range(0, n_total, batch):
                idx = list(range(s0, min(s0 + batch, n_total)))
                pending, tasks = {}, []
                for i in idx:
                    p = eng.pairs(icrs_to_array_rotation(t_first[i], float(theta_first[i])))
                    R0 = xp.asarray(icrs_to_array_rotation(t_mid[i], 0.0))  # true sky alt/az for the photometry
                    fb, pb = eng.ota_band[p.full_ota], eng.ota_band[p.part_ota]
                    rows_b, alt, az, filt = {}, [], [], []
                    for b, band in enumerate(eng.bands):
                        rb = xp.unique(xp.concatenate([p.full_row[fb == b], p.part_row[pb == b]]))
                        if len(rb) == 0:
                            continue
                        V = eng.T[rb] @ R0.T
                        rows_b[b] = rb
                        alt.append(h(xp.degrees(xp.arcsin(xp.clip(V[:, 2], -1, 1)))))
                        az.append(h(xp.degrees(xp.arctan2(V[:, 1], V[:, 0])) % 360.0))
                        filt.append(np.full(len(rb), band))
                    grid_tab = tbl.Table(
                        {"alt": np.concatenate(alt), "az": np.concatenate(az), "filter": np.concatenate(filt)}
                    )
                    pending[i] = (p, rows_b, icrs_to_array_rotation(t_first[i], float(theta_first[i])))
                    tasks.append(
                        (
                            i,
                            t_mid[i],
                            band_params,
                            seeing_mean,
                            seeing_std,
                            None,
                            int(seeds[i]),
                            int(n_dark[i]),
                            grid_tab,
                            float(solar[i]),
                        )
                    )
                for i, res in pool.imap(_process_ratchet_mp, tasks):
                    n_done += 1
                    if current_night is not None and nights[i] != current_night:
                        finish_night(current_night)
                    current_night = nights[i]
                    p, rows_b, R = pending.pop(i)
                    if res is None:
                        continue
                    # Per (tile, band) row from the worker: background variance of one exposure per pixel, full-
                    # system throughput, airmass, moon mask.  Rows come back in rows_b order.
                    bkg_all = np.asarray(res["bkg_e"], float)
                    tp_all = np.asarray(res["signal_tp"], float)
                    X_all = np.asarray(res["airmass"], float)
                    m_all = np.asarray(res["masked"], bool)
                    seg_of, off = {}, 0
                    masked_tile = np.zeros(eng.n_tiles, bool)
                    for b, rb in rows_b.items():
                        rb_h = h(rb)
                        seg_of[b] = (rb_h, slice(off, off + len(rb_h)))
                        masked_tile[rb_h[m_all[off : off + len(rb_h)]]] = True
                        off += len(rb_h)
                    # moon-masked tiles are not observed: drop their pairs before anything is summed
                    mt = xp.asarray(masked_tile)
                    kf, kp = ~mt[p.full_row], ~mt[p.part_row]
                    n_masked = int(h((~kf).sum())) + int(h((~kp).sum()))
                    p = RatchetPairs(
                        full_row=p.full_row[kf],
                        full_ota=p.full_ota[kf],
                        part_row=p.part_row[kp],
                        part_ota=p.part_ota[kp],
                        part_mask=p.part_mask[kp],
                        n_tiles_in_view=p.n_tiles_in_view,
                    )
                    # Full-system weight per covering pair: T^2 x sharpness / B, the sharpness at the
                    # pair's own field angle in its OTA and at the seeing at the tile's airmass (zenith draw x X^0.6).
                    fa_full, fa_part = (h(a) for a in eng.pair_field_angles(p, R))
                    fb, pb = h(eng.ota_band[p.full_ota]), h(eng.ota_band[p.part_ota])
                    fr_h, pr_h = h(p.full_row), h(p.part_row)
                    pair_noise = (np.full(len(fr_h), np.nan), np.full(len(pr_h), np.nan))
                    pair_tp = (np.full(len(fr_h), np.nan), np.full(len(pr_h), np.nan))
                    pair_X = (np.full(len(fr_h), np.nan), np.full(len(pr_h), np.nan))
                    inv = {}
                    for b, (rb_h, sl) in seg_of.items():
                        band = eng.bands[b]
                        B, T, X = bkg_all[sl], tp_all[sl], X_all[sl]
                        seeing_row = seeing_at_airmass(res["seeing"], X)
                        ws = []
                        for k, (rows_p, sel_b, fa) in enumerate(((fr_h, fb == b, fa_full), (pr_h, pb == b, fa_part))):
                            pos = np.searchsorted(rb_h, rows_p[sel_b])
                            sh_ = sharpness_at(tables[band], fa[sel_b], seeing_row[pos])
                            ok = B[pos] > 0
                            w = np.where(ok, T[pos] ** 2 * sh_ / np.where(ok, B[pos], 1.0), 0.0)
                            pair_noise[k][sel_b] = np.sqrt(B[pos] / sh_)
                            pair_tp[k][sel_b] = T[pos]
                            pair_X[k][sel_b] = X[pos]
                            ws.append(xp.asarray(w))
                        inv[b] = tuple(ws)
                    fast = res["exptime_s"] < cfg.survey.base_cadence_s
                    night_moon.append(float(res.get("moon_frac", np.nan)))
                    night_fast.append(bool(fast))
                    night_atm.append((res["seeing"], res["transparency"], res["solar_activity"], n_masked))
                    acc.add(p, inv, int(res["n_epochs"]), fast=fast)
                    write_epoch_record(
                        os.path.join(outdir, "epochs", f"ratchet_{i:07d}.npz"),
                        eng,
                        p,
                        pair_noise,
                        pair_tp,
                        dict(
                            ratchet=i,
                            mjd_first=t_first.mjd[i],
                            mjd_mid=t_mid.mjd[i],
                            theta_deg=theta_first[i],
                            n_epochs=int(res["n_epochs"]),
                            exptime_s=float(res["exptime_s"]),
                            night=int(nights[i]),
                            seeing_zenith=float(res["seeing"]),
                            transparency=float(res["transparency"]),
                            solar_activity=float(res["solar_activity"]),
                            field_angle=np.concatenate([fa_full, fa_part]).astype(np.float32),
                            airmass=np.concatenate(pair_X).astype(np.float32),
                        ),
                    )
                sample_gpu()
                self.log.info(f"Progress: {n_done}/{n_total} ratchets ({100 * n_done / n_total:.1f}%)")
            if current_night is not None:
                finish_night(current_night)
        finally:
            pool.close()
            pool.join()
        manifest = {
            "coverage": "exact",
            "backend": xp.__name__,
            "tessellation": f"HDPS tanseg/minipix via skymap_shim (panoptes {PANOPTES_COMMIT})",
            "bands": eng.bands,
            "n_ratchets": int(n_total),
            "band_zeropoints": band_zeropoints,
            "depth_convention": (
                "full system: exact eta(X) (optics x QE x atmosphere^X under the filter) x grey transparency draw; "
                "PSF sharpness per covering pair at its field angle and at seeing x X^0.6; moon-masked tiles dropped"
            ),
            "psf_model": cfg.telescope.psf_model,
            "psf_fits_path": cfg.telescope.psf_fits_path,
            "ee50_profile_path": cfg.telescope.ee50_profile_path,
            "transparency_prior": {
                "dist": transparency_prior(DEFAULT_RATCHET_PRIORS).dist,
                **transparency_prior(DEFAULT_RATCHET_PRIORS).params,
            },
            "solar_activity_prior": cfg.survey.solar_activity_prior,
            "gpu_pool_fraction": cfg.survey.gpu_pool_fraction,
            "gpu_device_used_peak_bytes": int(gpu["peak_used"]),
            "gpu_device_total_bytes": gpu["total"],
            "nights": nights_out,
        }
        with open(os.path.join(outdir, "exact_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1, default=float)
        return manifest

    def _exact_night_summary(self, eng, acc, owned, band_zeropoints, chunk: int = 4096) -> dict:
        """Summarise the night per band over its covered owned minipixes.

        Full-system limiting magnitude median / p16 / p84, and n_obs (all, base-cadence, fast-cadence).
        """
        from .tanseg_coverage import FAST_UNIT

        cfg = self._config
        xp = eng.xp
        rows = xp.nonzero(acc.touched)[0]
        out = {}
        for b, band in enumerate(eng.bands):
            zp = band_zeropoints[band]["zp_photons_per_sec"]
            qe = band_zeropoints[band]["band_qe"]
            mags, nobs, nobs60, nobs1 = [], [], [], []
            for s in range(0, len(rows), chunk):
                r = rows[s : s + chunk]
                iv = acc.ivar[b, r].astype(xp.float64)
                n60 = acc.nobs[b, r].astype(xp.float64)
                n1 = acc.fast_units[b, r].astype(xp.float64) * FAST_UNIT if acc.fast_units is not None else 0.0 * n60
                n = n60 + n1
                ok = (iv > 0) & (n > 0) & owned[r]
                exp_per = (n60 * cfg.survey.base_cadence_s + n1 * cfg.survey.fast_cadence_s)[ok] / n[ok]
                rate = cfg.survey.detection_snr / xp.sqrt(iv[ok]) / exp_per / qe
                # quantiles on the host: a device sort of ~10^8 values needs memory outside CuPy's pool
                mag = (-2.5 * xp.log10(rate / zp)).astype(xp.float32)
                mags.append(mag.get() if xp is not np else mag)
                nn = n[ok].astype(xp.float32)
                nobs.append(nn.get() if xp is not np else nn)
                b60 = n60[ok].astype(xp.float32)
                b1 = (n1 if not isinstance(n1, float) else 0.0 * n60)[ok].astype(xp.float32)
                nobs60.append(b60.get() if xp is not np else b60)
                nobs1.append(b1.get() if xp is not np else b1)
            m = np.concatenate(mags) if mags else np.zeros(0, np.float32)
            n = np.concatenate(nobs) if nobs else np.zeros(0, np.float32)
            if len(m) == 0:
                continue
            q = [float(v) for v in np.percentile(m, [16, 50, 84])]
            out.update(
                {
                    f"{band}_n_minipix": int(len(m)),
                    f"{band}_median_limmag": q[1],
                    f"{band}_p16_limmag": q[0],
                    f"{band}_p84_limmag": q[2],
                    f"{band}_median_n_obs": float(np.median(n)),
                    f"{band}_mean_n_obs": float(n.mean()),
                    f"{band}_median_n_obs_base": float(np.median(np.concatenate(nobs60))),
                    f"{band}_median_n_obs_fast": float(np.median(np.concatenate(nobs1))),
                }
            )
        return out

    def run(self, seed=None):
        """Run the survey simulation."""
        cfg = self._config
        rng = np.random.default_rng(seed)
        # Ratchets follow the exact schedule: ratchet_len-minute blocks on UTC boundaries,
        # tracked for ratchet_len - reset_min minutes at the sidereal rate, then reset with no exposures.
        from .coverage_exact import Schedule

        schedule = Schedule(
            exptime_s=cfg.survey.base_cadence_s,
            track_min=cfg.survey.ratchet_len - cfg.survey.reset_min,
            reset_min=cfg.survey.reset_min,
            sun_alt_max_deg=cfg.survey.min_sun_alt,
        )
        ex = schedule.exposures(self.start, self.end, self.obs.el)
        blocks, first, n_dark = np.unique(ex["ratchet"], return_index=True, return_counts=True)
        last = first + n_dark - 1
        ratchet_start_list = ex["time"][first]  # first dark exposure's mid-time: sets the geometry
        ratchet_mid_list = atime.Time((ex["time"].mjd[first] + ex["time"].mjd[last]) / 2, format="mjd", scale="utc")
        theta_first = ex["theta_deg"][first]
        # night index: local noon to local noon
        nights = np.floor(ratchet_start_list.mjd + cfg.observatory.longitude / 360.0 - 0.5).astype(int)
        keep_idx = np.arange(len(blocks))

        if cfg.survey.n_healpix is not None:
            hpx_to_keep = rng.choice(np.arange(self.hpx.npix), cfg.survey.n_healpix, replace=False)
            self.log.info(f"Sampling {len(hpx_to_keep)} healpix from the skymap.")
            pd.DataFrame(
                {
                    "healpix": hpx_to_keep,
                    "NSIDE": [
                        self.hpx.nside,
                    ]
                    * len(hpx_to_keep),
                    "nested": [
                        1,
                    ]
                    * len(hpx_to_keep),
                }
            ).to_parquet(f"{cfg.output.output_dir}/hpx_sample.parquet", compression="lz4")
        else:
            hpx_to_keep = None

        if cfg.survey.apply_weather:
            self.log.info("Starting nightly weather cull.")
            night_list = np.array(list(set(nights)))
            self.log.info(f"Found {len(night_list)} in survey plan.")
            night_starts = []
            for night in night_list:
                night_starts.append(ratchet_start_list[nights == night][0])
            nights_to_keep = night_list[[observable_weather(t, rng=rng) for t in night_starts]]
            self.log.info(f"Keeping {len(nights_to_keep)} out of {len(night_list)} nights.")
            nights_mask = np.array([np.isin(n, nights_to_keep) for n in nights])
            self.log.info(f"Keeping {nights_mask.sum()} ratchets (mask).")

            ratchet_start_list = ratchet_start_list[nights_mask]
            nights = nights[nights_mask]
            keep_idx = keep_idx[nights_mask]

        # Randomly select a fraction of the ratchets to run to account for
        # inter-night failures, filter-swap downtimes, etc.
        n_ratchets_initial = len(ratchet_start_list)

        n_keep = int(np.round(len(keep_idx) * cfg.survey.fraction_of_ratchets_to_run))
        sel = np.sort(rng.choice(len(keep_idx), n_keep, replace=False))
        ratchet_start_list, nights, keep_idx = ratchet_start_list[sel], nights[sel], keep_idx[sel]
        ratchet_mid_list = ratchet_mid_list[keep_idx]
        n_dark = n_dark[keep_idx]
        theta_first = theta_first[keep_idx]
        n_ratchets = len(ratchet_start_list)

        self.log.info(
            f"Running {n_ratchets} / {n_ratchets_initial} ratchets ({cfg.survey.fraction_of_ratchets_to_run * 100: 0.0f}% usable)."
        )

        if len(ratchet_start_list) == 0:
            self.log.error(
                f"No ratchets to run. Check survey dates or try increasing config.survey.fraction_of_ratchets_to_run from {cfg.survey.fraction_of_ratchets_to_run}."
            )
            return None

        if cfg.observatory.seasonal_seeing:
            seeing_mean = None
            seeing_std = None
        else:
            seeing_mean = cfg.observatory.seeing_mean
            seeing_std = cfg.observatory.seeing_std

        bands = sorted(set(cfg.filter_strategy.options))
        band_params = build_band_params(bands, self.throughputs, self.ab)
        for band, bp in band_params.items():
            self.log.info(
                f"Band {band}: signal_tp_no_atm={bp['signal_tp_no_atm']:.4f}, "
                f"atm_tp_am1={bp['atm_tp_am1']:.4f}, "
                f"airglow={bp['sky'].e_per_s10_airglow:.4e}, "
                f"zodiacal={bp['sky'].e_per_s10_zodiacal:.4e}"
            )

        ratchet_nums = np.arange(len(ratchet_start_list))
        mjds = np.array([t.mjd for t in ratchet_start_list])
        registry = build_window_registry(ratchet_nums, mjds, nights, cfg.output.cadence_levels)
        mjd_map = {int(r): float(m) for r, m in zip(ratchet_nums, mjds)}

        band_zeropoints = {}
        for band, bp in band_params.items():
            zp_photons = self.ab.photons_from_mag(band, 0.0)
            band_zeropoints[band] = {
                "zp_photons_per_sec": float(zp_photons.value),
                "band_qe": float(bp["band_qe"].value if hasattr(bp["band_qe"], "value") else bp["band_qe"]),
            }

        accumulator = CadenceAccumulator(
            window_registry=registry,
            cadence_levels=cfg.output.cadence_levels,
            nside_sparse=self.hpx.nside,
            output_dir=cfg.output.output_dir,
            mjds=mjd_map,
            band_zeropoints=band_zeropoints,
        )

        ratchet_seeds = rng.integers(0, 2**63, size=len(ratchet_start_list))
        if cfg.survey.solar_activity_prior == "epoch":
            solar = solar_activity_by_ratchet(ratchet_mid_list, nights, seed=seed)
        elif cfg.survey.solar_activity_prior == "fixed":
            solar = np.full(len(ratchet_start_list), cfg.observatory.solar_activity)
        else:
            raise ValueError(
                f"survey.solar_activity_prior must be 'epoch' or 'fixed', not {cfg.survey.solar_activity_prior!r}"
            )
        if cfg.survey.coverage == "exact":
            return self._run_exact(
                ratchet_start_list,
                ratchet_mid_list,
                theta_first,
                n_dark,
                nights,
                ratchet_seeds,
                band_params,
                seeing_mean,
                seeing_std,
                band_zeropoints,
                solar=solar,
            )
        if cfg.survey.coverage != "moc":
            raise ValueError(f"survey.coverage must be 'exact' or 'moc', not {cfg.survey.coverage!r}")
        args = [
            (
                ratchet_num,
                t,
                band_params,
                seeing_mean,
                seeing_std,
                hpx_to_keep,
                int(ratchet_seeds[i]),
                int(n_dark[i]),
                None,
                float(solar[i]),
            )
            for i, (ratchet_num, t) in enumerate(enumerate(ratchet_mid_list))
        ]

        n_total = len(args)
        n_done = 0
        with mp.Pool(
            cfg.nproc,
            initializer=_init_worker,
            initargs=(self.cradle, cfg, self.sharpness_tables()),
            maxtasksperchild=200,
        ) as pool:
            for ratchet_num, result in pool.imap(_process_ratchet_mp, args):
                n_done += 1
                if result is not None:
                    accumulator.ingest(ratchet_num, result, n_epochs=result["n_epochs"])
                else:
                    accumulator.skip(ratchet_num)
                if n_done % 200 == 0 or n_done == n_total:
                    self.log.info(f"Progress: {n_done}/{n_total} ratchets ({100 * n_done / n_total:.1f}%)")

        manifest = accumulator.finalize()
        manifest["coverage"] = "moc"
        return manifest
