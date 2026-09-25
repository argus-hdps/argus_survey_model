"""Run a Survey over an arbitrary span and summarise every night and the span.

Used by ``asim survey``.  Defaults come from the configuration: the exact coverage engine, the configured ratchet
and reset, the configured ring layout and filter cycle.  With ``local_scratch`` the run writes to a directory on
local disk and is copied to ``outdir`` afterwards: the per-night flush of the minipix totals can be ~30x slower on
NFS (69 s against 2.4 s per night in one test).
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import shutil
import tempfile
import time

import numpy as np

from argus_sim import c, get_logger

log = get_logger(__name__)


def resolve_span(start: str, end: str | None = None, n_nights: int | None = None) -> tuple[dt.datetime, dt.datetime]:
    """(start, end) as UTC datetimes from an ISO start and either an ISO end or a number of nights (24-h days)."""
    if (end is None) == (n_nights is None):
        raise ValueError("give exactly one of end or n_nights")
    t0 = dt.datetime.fromisoformat(start)
    if n_nights is not None:
        if n_nights < 1:
            raise ValueError(f"n_nights must be >= 1, got {n_nights}")
        t1 = t0 + dt.timedelta(days=int(n_nights))
    else:
        t1 = dt.datetime.fromisoformat(end)
    if t1 <= t0:
        raise ValueError(f"end {t1.isoformat()} is not after start {t0.isoformat()}")
    return t0, t1


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of ``asim survey``."""
    ap = argparse.ArgumentParser(description="Run an ArgusSim Survey over a span and summarise every night.")
    ap.add_argument("--start", required=True, help="UTC start, ISO (e.g. 2026-06-14T12:00)")
    span = ap.add_mutually_exclusive_group(required=True)
    span.add_argument("--end", help="UTC end, ISO")
    span.add_argument("--n-nights", type=int, help="number of 24-h days from --start")
    ap.add_argument("--outdir", required=True, help="final output directory")
    ap.add_argument(
        "--local-scratch",
        default=None,
        help="run in a new directory under this local path, then copy to --outdir (recommended on NFS)",
    )
    ap.add_argument("--coverage", choices=("exact", "moc"), default=None, help="default: config (exact)")
    ap.add_argument("--layout", default=None, help="ring layout name or path; default: config (A170r_1200)")
    ap.add_argument("--cycle", default=None, help="filter cycle, comma-separated; default: config")
    ap.add_argument("--seed", type=int, default=None, help="Survey seed; drawn and recorded if omitted")
    ap.add_argument("--nproc", type=int, default=None, help="CPU workers for photometry; default: config")
    ap.add_argument("--sim-depth", type=int, default=None, help="HEALPix depth (moc engine only)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument(
        "--weather",
        dest="weather",
        action="store_true",
        default=None,
        help="apply the nightly weather cull (default: config, on)",
    )
    g.add_argument(
        "--no-weather", dest="weather", action="store_false", help="observe every night (deterministic span)"
    )
    ap.add_argument("--no-record-stats", action="store_true", help="skip per-night swap / n_looks / edge stats")
    return ap


def configure(args: argparse.Namespace, workdir: str) -> dict:
    """Apply the arguments to the global config; return the resolved run settings."""
    t0, t1 = resolve_span(args.start, args.end, args.n_nights)
    s = c.survey
    s.start_date, s.end_date = t0, t1
    if args.coverage is not None:
        s.coverage = args.coverage
    if args.layout is not None:
        c.packing_strategy.layout_file = args.layout
    if args.cycle is not None:
        c.filter_strategy.options = args.cycle.split(",")
    if args.nproc is not None:
        c.nproc = args.nproc
    if args.sim_depth is not None:
        s.sim_depth = args.sim_depth
    if args.weather is not None:
        s.apply_weather = bool(args.weather)
    c.output.output_dir = workdir
    c.output.cadence_levels = ["ratchet", "night"]
    c.output.save_grid_tables = True
    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().generate_state(1)[0] >> 1)
    return {
        "start": t0.isoformat(),
        "end": t1.isoformat(),
        "coverage": s.coverage,
        "ratchet_len": s.ratchet_len,
        "reset_min": s.reset_min,
        "cycle": list(c.filter_strategy.options),
        "seed": int(seed),
        "nproc": c.nproc,
        "apply_weather": s.apply_weather,
    }


def night_table(manifest: dict) -> list[dict]:
    """One row per night and band from an exact_manifest.json."""
    rows = []
    for nt in manifest["nights"]:
        for band in manifest["bands"]:
            if f"{band}_n_minipix" not in nt:
                continue
            rows.append(
                {
                    "night": nt["night"],
                    "mjd_min": nt["mjd_min"],
                    "mjd_max": nt["mjd_max"],
                    "band": band,
                    "n_minipix": nt[f"{band}_n_minipix"],
                    "area_deg2": nt.get(f"{band}_area_deg2"),
                    "median_limmag": nt[f"{band}_median_limmag"],
                    "p16_limmag": nt[f"{band}_p16_limmag"],
                    "p84_limmag": nt[f"{band}_p84_limmag"],
                    "median_n_obs": nt[f"{band}_median_n_obs"],
                    "mean_n_obs": nt[f"{band}_mean_n_obs"],
                    "median_n_obs_base": nt.get(f"{band}_median_n_obs_base"),
                    "median_n_obs_fast": nt.get(f"{band}_median_n_obs_fast"),
                    "n_ratchets": nt.get("n_ratchets"),
                    "moon_frac_min": nt.get("moon_frac_min"),
                    "moon_frac_median": nt.get("moon_frac_median"),
                    "moon_frac_max": nt.get("moon_frac_max"),
                    "n_fast_ratchets": nt.get("n_fast_ratchets"),
                }
            )
    return rows


def span_summary(workdir: str, manifest: dict, chunk: int = 2048) -> dict:
    """Depth and coverage of the whole span from the minipix totals (owned minipixes only)."""
    from .coverage_exact import get_backend
    from .tanseg_coverage import MINIPIX_AREA_DEG2, owned_mask

    xp = get_backend("auto")
    d = os.path.join(workdir, "minipix")
    tid = np.load(os.path.join(d, "tanseg_id.npy"))
    ivar = np.load(os.path.join(d, "total_ivar.npy"), mmap_mode="r")
    nobs = np.load(os.path.join(d, "total_nobs.npy"), mmap_mode="r")
    from .tanseg_coverage import FAST_UNIT

    fast_path = os.path.join(d, "total_fast_units.npy")
    nfast = np.load(fast_path, mmap_mode="r") if os.path.exists(fast_path) else None
    zps = manifest["band_zeropoints"]
    snr, t60, t1 = c.survey.detection_snr, c.survey.base_cadence_s, c.survey.fast_cadence_s
    out = {}
    for b, band in enumerate(manifest["bands"]):
        mags, ns = [], []
        for s in range(0, len(tid), chunk):
            sl = slice(s, s + chunk)
            n60 = np.asarray(nobs[b, sl], np.float64)
            n1 = np.asarray(nfast[b, sl], np.float64) * FAST_UNIT if nfast is not None else 0.0 * n60
            if not (n60.any() or (nfast is not None and n1.any())):
                continue
            n = n60 + n1
            iv = np.asarray(ivar[b, sl], np.float64)
            own = owned_mask(tid[sl], xp)
            ok = (iv > 0) & (n > 0) & (own.get() if xp is not np else own)
            exp_per = (n60 * t60 + n1 * t1)[ok] / n[ok]
            rate = snr / np.sqrt(iv[ok]) / exp_per / zps[band]["band_qe"]
            mags.append((-2.5 * np.log10(rate / zps[band]["zp_photons_per_sec"])).astype(np.float32))
            ns.append(n[ok].astype(np.float32))
        m, n = np.concatenate(mags), np.concatenate(ns)
        out[band] = {
            "n_minipix": int(len(m)),
            "area_deg2": float(len(m) * MINIPIX_AREA_DEG2),
            "median_limmag": float(np.median(m)),
            "p16_limmag": float(np.percentile(m, 16)),
            "p84_limmag": float(np.percentile(m, 84)),
            "median_n_obs": float(np.median(n)),
            "mean_n_obs": float(n.mean()),
        }
    return out


def record_stats(workdir: str, n_sample_tiles: int = 4000, seed: int = 5) -> list[dict]:
    """Per-night statistics from the per-epoch records.

    Pointing-to-pointing band swap, seam share, both-band share, same-band n_looks share (on a tile sample), and
    the southern edge in of-date declination.
    """
    import astropy.time as atime
    import erfa

    from . import skymap_shim as sh

    recs = sorted(glob.glob(os.path.join(workdir, "epochs", "ratchet_*.npz")))
    by_night: dict[int, list[str]] = {}
    for f in recs:
        with np.load(f) as d:
            by_night.setdefault(int(d["night"]), []).append(f)
    xi, eta = sh.minipix_offsets()
    rng = np.random.default_rng(seed)
    out = []
    for night, files in sorted(by_night.items()):
        labels, looks_ge2, looks_cov, south, sample = [], 0, 0, [], None
        for f in files:
            d = np.load(f)
            tids, band, pf = d["tanseg_id"], d["band"], int(d["partial_from"])
            pm = np.unpackbits(d["partial_mask"], axis=1)[:, : sh.N_MINIPIX].astype(bool)
            t = atime.Time(float(d["mjd_first"]), format="mjd", scale="utc")
            M = erfa.c2i06a(t.tt.jd1, t.tt.jd2)
            ra_c, dec_c = sh.tile_center(tids)
            s = np.flatnonzero(dec_c < -20)
            if len(s):
                T, E, N = sh.tangent_basis(ra_c[s], dec_c[s])
                P = T[:, None, :] + xi[None, :, None] * E[:, None, :] + eta[None, :, None] * N[:, None, :]
                P /= np.linalg.norm(P, axis=-1, keepdims=True)
                dec = np.degrees(np.arcsin((P @ M.T)[..., 2]))
                m = np.ones((len(s), sh.N_MINIPIX), bool)
                part = s >= pf
                m[part] = pm[s[part] - pf]
                south.append(float(dec[m].min()))
            if sample is None:
                u = np.unique(tids)
                sample = np.sort(rng.choice(u, min(n_sample_tiles, len(u)), replace=False))
            lab = np.zeros((len(sample), sh.N_MINIPIX), np.int8)
            cnt = np.zeros((2, len(sample), sh.N_MINIPIX), np.int16)
            pos = np.searchsorted(sample, tids)
            ok = (pos < len(sample)) & (sample[np.clip(pos, 0, len(sample) - 1)] == tids)
            for k in np.flatnonzero(ok):
                mk = np.ones(sh.N_MINIPIX, bool) if k < pf else pm[k - pf]
                lab[pos[k]] |= np.where(mk, 1 << int(band[k]), 0).astype(np.int8)
                cnt[min(int(band[k]), 1), pos[k]] += mk
            looks_cov += int((cnt.sum(0) > 0).sum())
            looks_ge2 += int((cnt >= 2).any(0).sum())
            labels.append(lab)
        L = np.stack(labels)
        a, b = L[:-1], L[1:]
        tr = (a > 0) & (b > 0)
        swap = tr & (((a == 1) & (b == 2)) | ((a == 2) & (b == 1)))
        pure = tr & (a < 3) & (b < 3)
        seen = (L > 0).any(0)
        out.append(
            {
                "night": night,
                "n_ratchets": len(files),
                "swap": float(swap.sum() / max(tr.sum(), 1)),
                "swap_single_band": float(swap.sum() / max(pure.sum(), 1)),
                "seam_pointings": float((L == 3).sum() / max((L > 0).sum(), 1)),
                "both_bands_in_night": float(((L & 1).any(0) & (L & 2).any(0)).sum() / max(seen.sum(), 1)),
                "same_band_looks_ge2": float(looks_ge2 / max(looks_cov, 1)),
                "southern_edge_cirs_min": float(min(south)) if south else None,
                "southern_edge_cirs_max": float(max(south)) if south else None,
            }
        )
    return out


def run(argv: list[str] | None = None) -> dict:
    """Parse, configure, run, summarise, and copy from local scratch if asked.  Returns the span summary."""
    args = build_parser().parse_args(argv)
    outdir = os.path.abspath(args.outdir)
    if args.local_scratch:
        os.makedirs(args.local_scratch, exist_ok=True)
        workdir = tempfile.mkdtemp(prefix="argussim_", dir=args.local_scratch)
    else:
        os.makedirs(outdir, exist_ok=True)
        workdir = outdir
    settings = configure(args, workdir)
    settings.update({"outdir": outdir, "workdir": workdir})
    log.info(
        f"Survey {settings['start']} -> {settings['end']}, coverage {settings['coverage']}, seed {settings['seed']}"
    )

    from .provenance import write_provenance
    from .survey import Survey

    t_start = time.time()
    peak_gpu = None
    survey = Survey()
    manifest = survey.run(seed=settings["seed"])
    if manifest is None:
        raise SystemExit("No ratchets survived culling")
    wall = time.time() - t_start
    try:
        import cupy as cp

        peak_gpu = cp.get_default_memory_pool().total_bytes()
    except Exception:
        pass

    summary: dict = {
        "settings": settings,
        "wall_s": wall,
        "gpu_pool_peak_bytes": peak_gpu,
        "gpu_device_used_peak_bytes": manifest.get("gpu_device_used_peak_bytes"),
        "gpu_pool_fraction": manifest.get("gpu_pool_fraction"),
        "depth_convention": manifest.get("depth_convention"),
    }
    if manifest.get("coverage") == "exact":
        rows = night_table(manifest)
        import pandas as pd

        pd.DataFrame(rows).to_csv(os.path.join(workdir, "night_summary.csv"), index=False)
        summary["nights"] = rows
        summary["span"] = span_summary(workdir, manifest)
        if not args.no_record_stats:
            stats = record_stats(workdir)
            pd.DataFrame(stats).to_csv(os.path.join(workdir, "night_record_stats.csv"), index=False)
            summary["night_record_stats"] = stats
    summary["disk_bytes"] = int(sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(workdir) for f in fs))
    with open(os.path.join(workdir, "span_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    write_provenance(workdir, "survey", settings)

    if workdir != outdir:
        t_copy = time.time()
        shutil.copytree(workdir, outdir, dirs_exist_ok=True)
        summary["copy_s"] = time.time() - t_copy
        with open(os.path.join(outdir, "span_summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=float)
        shutil.rmtree(workdir)
        log.info(f"Copied {workdir} -> {outdir} in {summary['copy_s']:.0f} s and removed the local copy.")
    return summary
