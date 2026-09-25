"""The visit store: a run's (tile, OTA) pair records sorted by tile, compact and self-contained.

A run is written in time order (``epochs/ratchet_NNNNNNN.npz``), but a light curve needs every record of one place.
``build_index`` makes the store in two stages.

1. Sort. Passes over contiguous tile ranges, each small enough to sort in memory, write the full-precision pair
   rows in tile order to ``build/sorted/``. Each pass re-reads the small record columns of every ratchet.
2. Compact. One worker per sorted shard writes ``pairs/shard_NNNN.parquet`` with four columns per pair row
   (ratchet, tel_id, partial flag, 16-bit log-quantised sigma_bkg_e), and the tile depth maps from the full
   records. Airmass, throughput, field angle and n_covered are left out: a query recomputes them from the ratchet
   rotations and OTA frames frozen into the store (``geometry``). The build recomputes them for every row and
   stops if any differs from the run's value by more than ``TOLERANCE``.

Finished shards are never rewritten, so an interrupted build resumes where it stopped. The store directory holds
everything a user needs (``index.json``, ``ratchets.npz``, ``layout.npz``, ``tiles.npz``, ``pairs/``,
``depth.parquet``, ``nights.parquet``, ``manifest.json``); ``build/`` is scratch space and is not part of it.
"""

from __future__ import annotations

import datetime
import glob
import hashlib
import json
import multiprocessing as mp
import os
import resource
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from argus_sim import get_logger

from .. import skymap_shim as sh
from . import depth as dp
from . import geometry as g

log = get_logger(__name__)

FORMAT_VERSION = 2
RATCHET_KEYS = (
    "ratchet",
    "mjd_first",
    "mjd_mid",
    "n_epochs",
    "exptime_s",
    "night",
    "theta_deg",
    "seeing_zenith",
    "transparency",
    "solar_activity",
)
SORTED_SCHEMA = pa.schema(
    [
        ("tanseg_id", pa.int32()),
        ("ratchet", pa.int32()),
        ("tel_id", pa.int16()),
        ("band", pa.int8()),
        ("partial", pa.bool_()),
        ("n_covered", pa.int16()),
        ("sigma_bkg_e", pa.float32()),
        ("throughput", pa.float32()),
        ("field_angle", pa.float32()),
        ("airmass", pa.float32()),
    ]
)
PAIR_SCHEMA = pa.schema(
    [
        ("ratchet", pa.int32()),
        ("tel_id", pa.int16()),
        ("partial", pa.bool_()),
        ("sigma_bkg_q", pa.uint16()),
    ]
)
# A row group holds one tile, or a run of consecutive small tiles with at least this many rows between them: one
# row group per tile would put ~2 kB of Parquet metadata on each tile, which is most of the store for a short run.
MIN_GROUP_ROWS = 8192
# Largest allowed difference between a recomputed or quantised column and the run's value (relative, except
# field_angle in degrees; n_covered must match exactly).
TOLERANCE = {"airmass": 1e-4, "field_angle": 1e-3, "throughput": 1e-5, "sigma_bkg_e": 1e-4}
# Bytes per pair row held while a sort pass runs: the columns (30) and some slack for the chunks in flight.
BYTES_PER_ROW_IN_PASS = 32
Q16_MAX = 65535


def shard_path(root: str, shard: int) -> str:
    """Return the path of one shard's Parquet file under ``root`` (the store, or ``build/sorted``)."""
    return os.path.join(root, "pairs", f"shard_{shard:04d}.parquet")


def quantise(x: np.ndarray, q: dict) -> np.ndarray:
    """Return 16-bit codes of positive values with the log quantisation ``q`` ({"ln_min", "ln_step"})."""
    c = np.rint((np.log(np.asarray(x, np.float64)) - q["ln_min"]) / q["ln_step"])
    return np.clip(c, 0, Q16_MAX).astype(np.uint16)


def dequantise(c: np.ndarray, q: dict) -> np.ndarray:
    """Return the values of 16-bit codes."""
    return np.exp(q["ln_min"] + np.asarray(c, np.float64) * q["ln_step"])


def log_quantisation(lo: float, hi: float) -> dict:
    """Return log quantisation constants covering [lo, hi] in 65536 steps."""
    a, b = np.log(lo), np.log(hi)
    pad = 1e-6 * max(b - a, 1e-3)
    step = (b - a + 2 * pad) / Q16_MAX
    return {"encoding": "uint16 code c, value exp(ln_min + c * ln_step)", "ln_min": a - pad, "ln_step": step}


def throughput_table(bands: list[str]) -> dict:
    """Return the signal-throughput tables of the configured system, as the survey builds them."""
    from .. import ABPhot, SystemThroughput, c
    from ..spectral_sky import build_band_params

    filters = {b: True for b in c.filter_strategy.options}
    tp = SystemThroughput(throughput_loss=c.telescope.throughput_loss, filters=filters)
    ab = ABPhot(collecting_area=c.telescope.collecting_area, throughput_loss=c.telescope.throughput_loss, throughput=tp)
    bp = build_band_params(bands, tp, ab)
    grid = np.asarray(bp[bands[0]]["signal_tp_table"][0], np.float64)
    return {
        "definition": "throughput = exp(interp(airmass, airmass_grid, log_tp[band])) x transparency",
        "airmass": grid.tolist(),
        "log_tp": [np.log(np.asarray(bp[b]["signal_tp_table"][1], np.float64)).tolist() for b in bands],
    }


# ---------------------------------------------------------------------- stage 1 workers
_WORKER: dict = {}


def _init_worker(state: dict) -> None:
    _WORKER.update(state)
    _WORKER["geom"] = g.Geometry(**state["geom"])


def _check_masks(d, R, partial_from: int, every: int = 7, max_rows: int = 3000) -> tuple[int, int]:
    """Recompute a sample of a ratchet's partial masks point by point. Returns (bits checked, bits that differ)."""
    geom = _WORKER["geom"]
    tid = d["tanseg_id"]
    rows = np.arange(partial_from, len(tid))[::every][:max_rows]
    if len(rows) == 0:
        return 0, 0
    ref = np.unpackbits(d["partial_mask"][rows - partial_from], axis=1).astype(bool)
    T, E, N = g.tile_frames(tid[rows])
    xi, eta = sh.minipix_offsets()
    Tn, En, Nn = T @ R.T, E @ R.T, N @ R.T
    bad = 0
    for s in range(0, len(rows), 500):
        sl = slice(s, s + 500)
        P = Tn[sl, None, :] + xi[None, :, None] * En[sl, None, :] + eta[None, :, None] * Nn[sl, None, :]
        P /= np.linalg.norm(P, axis=-1, keepdims=True)
        m = geom.inside(P, geom.ota(d["tel_id"][rows[sl]])[:, None])
        bad += int((m != ref[sl]).sum())
    return int(ref.size), bad


def _census(args):
    path, check = args
    with np.load(path) as d:
        scal = {k: d[k].item() for k in RATCHET_KEYS}
        tid = d["tanseg_id"]
        pf = int(d["partial_from"])
        u, cnt = np.unique(tid, return_counts=True)
        R = g.ratchet_rotation(scal["mjd_first"], scal["theta_deg"])
        Rm = g.ratchet_rotation(scal["mjd_mid"], 0.0)
        chk = _check_masks(d, R, pf) if check else (0, 0)
        tb = np.unique(d["tel_id"].astype(np.int64) * 256 + d["band"].astype(np.int64))
        noise = d["noise"]
        nrange = (float(noise.min()), float(noise.max())) if len(noise) else (np.inf, -np.inf)
    return scal, R, Rm, u, cnt, len(tid) - pf, chk, tb, nrange


def _extract(args):
    path, lo, hi = args
    with np.load(path) as d:
        tid = d["tanseg_id"]
        sel = np.flatnonzero((tid >= lo) & (tid <= hi))
        if len(sel) == 0:
            return None
        return dict(
            tanseg_id=tid[sel],
            ratchet=np.full(len(sel), int(d["ratchet"]), np.int32),
            tel_id=d["tel_id"][sel],
            band=d["band"][sel],
            partial=sel >= int(d["partial_from"]),
            n_covered=d["n_covered"][sel],
            sigma_bkg_e=d["noise"][sel],
            throughput=d["signal_tp"][sel],
            field_angle=d["field_angle"][sel],
            airmass=d["airmass"][sel],
        )


# ---------------------------------------------------------------------- stage 2 worker
def _compact(shard: int) -> dict:
    """Write one compact shard and its depth-map rows from a sorted shard. Returns the shard's summary."""
    W = _WORKER
    out = shard_path(W["index_dir"], shard)
    side = os.path.join(W["build_dir"], "compact", f"shard_{shard:04d}.npz")
    if os.path.isfile(out) and os.path.isfile(side):
        with np.load(side, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    geom = W["geom"]
    rat = W["ratchet"]
    pf = pq.ParquetFile(shard_path(W["sorted_dir"], shard))
    tmp = out + ".tmp"
    writer = pq.ParquetWriter(
        tmp,
        PAIR_SCHEMA,
        compression="zstd",
        compression_level=6,
        use_dictionary=["tel_id"],
        column_encoding={"ratchet": "DELTA_BINARY_PACKED", "partial": "RLE", "sigma_bkg_q": "BYTE_STREAM_SPLIT"},
        write_statistics=False,
    )
    buf, buf_rows, group = [], 0, 0
    tiles = {k: [] for k in ("tanseg_id", "row_group", "offset", "n_rows")}
    depth_rows = []
    worst = {k: 0.0 for k in TOLERANCE}
    n_cov_bad = 0
    n_rows = 0

    def flush():
        nonlocal buf, buf_rows, group
        cols = {k: np.concatenate([b[k] for b in buf]) for k in PAIR_SCHEMA.names}
        writer.write_table(
            pa.Table.from_arrays([pa.array(cols[f.name], f.type) for f in PAIR_SCHEMA], schema=PAIR_SCHEMA),
            row_group_size=buf_rows + 1,
        )
        buf, buf_rows, group = [], 0, group + 1

    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg)
        d = {k: t.column(k).to_numpy() for k in t.column_names}
        tid = int(d["tanseg_id"][0])
        res = _tile(tid, d, geom, rat)
        for k in TOLERANCE:
            worst[k] = max(worst[k], res["worst"][k])
        n_cov_bad += res["n_cov_bad"]
        depth_rows.extend(res["depth"])
        n = len(d["ratchet"])
        tiles["tanseg_id"].append(tid)
        tiles["row_group"].append(group)
        tiles["offset"].append(buf_rows)
        tiles["n_rows"].append(n)
        buf.append(res["pairs"])
        buf_rows += n
        n_rows += n
        if buf_rows >= MIN_GROUP_ROWS:
            flush()
    if buf:
        flush()
    writer.close()
    dcols = list(zip(*depth_rows)) if depth_rows else [[] for _ in range(9)]
    summary = dict(
        shard=np.int64(shard),
        n_rows=np.int64(n_rows),
        n_cov_bad=np.int64(n_cov_bad),
        **{f"worst_{k}": np.float64(v) for k, v in worst.items()},
        **{f"tile_{k}": np.asarray(v, np.int64) for k, v in tiles.items()},
        depth_band=np.asarray(dcols[0], np.int8),
        depth_period=np.asarray(dcols[1], np.int16),
        depth_tanseg_id=np.asarray(dcols[2], np.int64),
        depth_limmag=np.asarray(dcols[3], np.float64),
        depth_covered_fraction=np.asarray(dcols[4], np.float64),
        depth_n_nights=np.asarray(dcols[5], np.int64),
        depth_n_slots_60=np.asarray(dcols[6], np.int64),
        depth_n_slots_1=np.asarray(dcols[7], np.int64),
        depth_n_frames=np.asarray(dcols[8], np.int64),
    )
    stmp = side + ".tmp.npz"
    np.savez(stmp, **summary)
    os.replace(tmp, out)
    os.replace(stmp, side)
    return summary


def _tile(tid: int, d: dict, geom: g.Geometry, rat: dict) -> dict:
    """Recompute, check and compact one tile's pair rows, and sum its depth maps."""
    from ..tanseg_coverage import owned_mask

    W = _WORKER
    n = len(d["ratchet"])
    rpos = np.searchsorted(rat["ratchet"], d["ratchet"])
    R, Rm = rat["rotation"][rpos], rat["rotation_mid"][rpos]
    T, E, N = (np.broadcast_to(v[0], (n, 3)) for v in g.tile_frames(tid))
    ota = geom.ota(d["tel_id"])
    band = W["band_of_tel"][d["tel_id"]]
    if np.any(band != d["band"]):
        raise RuntimeError(f"Tile {tid}: a pair's band is not its OTA's band in the layout.")
    X = g.airmass(Rm, T)
    thr = g.throughput(W["tp_table"], band, X, rat["transparency"][rpos])
    fa = np.empty(n)
    full = ~d["partial"]
    fa[full] = g.field_angle(geom, R[full], T[full], E[full], N[full], ota[full])
    sig = d["sigma_bkg_e"].astype(np.float64)
    tp = d["throughput"].astype(np.float64)
    code = quantise(sig, W["quant"])
    worst = {
        "airmass": float(np.max(np.abs(X / d["airmass"] - 1), initial=0)),
        "throughput": float(np.max(np.abs(thr / tp - 1), initial=0)),
        "sigma_bkg_e": float(np.max(np.abs(dequantise(code, W["quant"]) / sig - 1), initial=0)),
    }
    n_epochs = rat["n_epochs"][rpos].astype(np.float64)
    exptime = rat["exptime_s"][rpos]
    fast = exptime < W["base_cadence_s"]
    weights = np.stack(
        [n_epochs * tp**2 / sig**2, np.where(fast, 0, n_epochs), np.where(fast, n_epochs, 0), n_epochs * exptime]
    )
    period = rat["period"][rpos]
    bp = band.astype(np.int64) * W["n_periods"] + period
    acc = dp.TileDepth(W["n_bands"], W["n_periods"])
    acc.add(bp[full], weights[:, full])
    n_cov_bad = 0
    part = np.flatnonzero(d["partial"])
    for c0 in range(0, len(part), 8000):
        p = part[c0 : c0 + 8000]
        lo, hi = g.row_runs(geom, R[p], T[p], E[p], N[p], ota[p])
        fa[p] = g.field_angle(geom, R[p], T[p], E[p], N[p], ota[p], runs=(lo, hi))
        n_cov_bad += int(np.sum(g.runs_count(lo, hi).sum(1) != d["n_covered"][p]))
        acc.add(bp[p], weights[:, p], lo, hi)
    n_cov_bad += int(np.sum(d["n_covered"][full] != sh.N_MINIPIX))
    worst["field_angle"] = float(np.max(np.abs(fa - d["field_angle"]), initial=0))

    n_slots = rat["n_slots"][rpos]
    counts = {}
    for key in np.unique(bp):
        sel = bp == key
        b, p = divmod(int(key), W["n_periods"])
        counts[(b, p)] = (
            len(np.unique(rat["night"][rpos[sel]])),
            int(n_slots[sel & ~fast].sum()),
            int(n_slots[sel & fast].sum()),
            int(rat["n_epochs"][rpos[sel]].sum()),
        )
    owned = np.asarray(owned_mask(np.array([tid])))[0].reshape(g.GRID, g.GRID)
    depth = dp.tile_rows(tid, acc.sums(), owned, counts, W["zeropoints"])
    pairs = dict(ratchet=d["ratchet"], tel_id=d["tel_id"], partial=d["partial"], sigma_bkg_q=code)
    return dict(pairs=pairs, depth=depth, worst=worst, n_cov_bad=n_cov_bad)


# ---------------------------------------------------------------------- build
def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _write_atomic_npz(path: str, compressed: bool = False, **arrays) -> None:
    tmp = path + ".tmp.npz"
    (np.savez_compressed if compressed else np.savez)(tmp, **arrays)
    os.replace(tmp, path)


def _write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _plan(tile_counts: np.ndarray, rows_per_pass: int, rows_per_shard: int, tiles_per_shard: int) -> dict:
    """Assign tiles (in tanseg order) to shards and shards to passes. No pass or shard splits a tile."""
    nz = np.flatnonzero(tile_counts)
    shard = np.empty(len(nz), np.int32)
    rg = np.empty(len(nz), np.int32)
    shard_pass = []
    s, p, in_shard, in_pass, k = 0, 0, 0, 0, 0
    for i, n in enumerate(tile_counts[nz]):
        if in_pass and in_pass + n > rows_per_pass:
            p, s, in_pass, in_shard, k = p + 1, s + 1, 0, 0, 0
        elif in_shard and (in_shard + n > rows_per_shard or k == tiles_per_shard):
            s, in_shard, k = s + 1, 0, 0
        if len(shard_pass) <= s:
            shard_pass.append(p)
        shard[i], rg[i] = s, k
        k += 1
        in_shard += n
        in_pass += n
    return dict(tile_pos=nz, shard=shard, row_group=rg, shard_pass=np.array(shard_pass, np.int32))


def _write_sorted_shard(path: str, cols: dict, tile_bounds: np.ndarray) -> None:
    tmp = path + ".tmp"
    with pq.ParquetWriter(tmp, SORTED_SCHEMA, compression="zstd", compression_level=1) as w:
        for a, b in zip(tile_bounds[:-1], tile_bounds[1:]):
            w.write_table(
                pa.Table.from_arrays(
                    [pa.array(cols[f.name][a:b], f.type) for f in SORTED_SCHEMA], schema=SORTED_SCHEMA
                ),
                row_group_size=int(b - a) + 1,
            )
    os.replace(tmp, path)


def build_index(
    run_dir: str,
    *,
    index_dir: str | None = None,
    workers: int | None = None,
    max_memory_gb: float = 4.0,
    rows_per_shard: int = 16_000_000,
    tiles_per_shard: int = 1024,
    layout: str | None = None,
    check_every: int = 500,
    limit_ratchets: int | None = None,
    tanseg_range: tuple[int, int] | None = None,
    keep_sorted: bool = False,
    sorted_from: str | None = None,
) -> str:
    """Build, or finish building, the visit store of ``run_dir`` and return the store directory.

    ``index_dir`` is the store directory (default ``run_dir/lightcurve``). ``max_memory_gb`` bounds the rows sorted
    in one pass (the process holds about this plus the worker pool). ``check_every``: every n-th ratchet's partial
    masks are compared bit by bit with the recomputed geometry. ``tiles_per_shard`` bounds a shard's footer.
    ``limit_ratchets`` indexes only the first n ratchet files (timing trials); ``tanseg_range`` = (lo, hi) only the
    tiles with lo <= tanseg_id <= hi. ``keep_sorted`` keeps the full-precision sorted rows in ``build/sorted``.
    ``sorted_from`` names a format-1 store of the same run (full-precision rows sorted by tile, one row group per
    tile), whose shards then replace the sort passes; they are read, never changed.
    """
    t_start = time.time()
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    index_dir = os.path.abspath(os.path.expanduser(index_dir or os.path.join(run_dir, "lightcurve")))
    workers = workers or min(4, os.cpu_count() or 1)
    files = sorted(glob.glob(os.path.join(run_dir, "epochs", "ratchet_*.npz")))
    if limit_ratchets is not None:
        files = files[:limit_ratchets]
    if not files:
        raise FileNotFoundError(f"No epochs/ratchet_*.npz files in {run_dir}.")
    build_dir = os.path.join(index_dir, "build")
    sorted_dir = os.path.join(build_dir, "sorted")
    if sorted_from is not None:
        if limit_ratchets is not None or tanseg_range is not None:
            raise ValueError("sorted_from cannot be combined with limit_ratchets or tanseg_range.")
        sorted_dir = os.path.abspath(os.path.expanduser(sorted_from))
    for d in (os.path.join(index_dir, "pairs"), os.path.join(sorted_dir, "pairs"), os.path.join(build_dir, "compact")):
        os.makedirs(d, exist_ok=True)
    with open(os.path.join(run_dir, "exact_manifest.json")) as f:
        manifest = json.load(f)
    prov_path = os.path.join(run_dir, "provenance.json")
    prov = json.load(open(prov_path)) if os.path.isfile(prov_path) else {}
    layout = layout or prov.get("ring_layout")
    if layout is None:
        raise ValueError("The run's provenance.json does not name its ring_layout. Pass layout=.")
    geom = g.Geometry.from_layout(layout)
    bands = [str(b) for b in manifest["bands"]]
    log.info(f"Building the visit store of {run_dir} ({len(files)} ratchets, layout {layout}) in {index_dir}.")

    # ---- stage 1: census and sort
    ctx = mp.get_context("forkserver")
    universe = sh.tiles()["tanseg_id"]
    census_path = os.path.join(build_dir, "census.npz")
    with ctx.Pool(workers, initializer=_init_worker, initargs=({"geom": geom.arrays()},)) as pool:
        if os.path.isfile(census_path) and len(np.load(census_path)["ratchet"]) == len(files):
            log.info("Reusing the census from an earlier build.")
            cz = dict(np.load(census_path))
        else:
            cz = _run_census(pool, files, universe, check_every)
            _write_atomic_npz(census_path, **cz)
        log.info(
            f"Census: {int(cz['tile_counts'].sum()):,} pair rows ({int(cz['n_partial'].sum()):,} partial) in "
            f"{int((cz['tile_counts'] > 0).sum()):,} tiles; mask check {int(cz['check_bad'])} of "
            f"{int(cz['check_bits']):,} bits differ."
        )
        if cz["check_bits"] and cz["check_bad"] > 1e-6 * cz["check_bits"]:
            raise RuntimeError(
                f"The recomputed partial masks differ from the run's in {int(cz['check_bad'])} of "
                f"{int(cz['check_bits'])} bits. The layout {layout!r} or the site in the configuration is not the run's."
            )
        rows_per_pass = int(max_memory_gb * 1e9 / BYTES_PER_ROW_IN_PASS)
        counts = cz["tile_counts"]
        if tanseg_range is not None:
            counts = np.where((universe >= tanseg_range[0]) & (universe <= tanseg_range[1]), counts, 0)
        plan = _plan(counts, rows_per_pass, rows_per_shard, tiles_per_shard)
        n_pass = int(plan["shard_pass"].max()) + 1
        n_shards = len(plan["shard_pass"])
        tile_tid = universe[plan["tile_pos"]]
        tile_n = counts[plan["tile_pos"]]
        log.info(
            f"Plan: {n_pass} sort passes of <= {rows_per_pass:,} rows, {n_shards} shards, {len(tile_tid):,} tiles."
        )

        if sorted_from is not None:
            n_shards = len(glob.glob(os.path.join(sorted_dir, "pairs", "shard_*.parquet")))
            if not all(os.path.isfile(shard_path(sorted_dir, s)) for s in range(n_shards)):
                raise FileNotFoundError(f"{sorted_dir}/pairs does not hold shards 0..{n_shards - 1}.")
            n_pass = 0
            log.info(f"Using the {n_shards} sorted shards of {sorted_dir} in place of the sort passes.")

        def shard_done(s):
            return os.path.isfile(shard_path(index_dir, s)) and os.path.isfile(
                os.path.join(build_dir, "compact", f"shard_{s:04d}.npz")
            )

        for p in range(n_pass):
            shards = np.flatnonzero(plan["shard_pass"] == p)
            if all(os.path.isfile(shard_path(sorted_dir, s)) or shard_done(s) for s in shards):
                log.info(f"Sort pass {p + 1}/{n_pass} was finished by an earlier build.")
                continue
            _run_pass(pool, files, p, n_pass, shards, plan, tile_tid, tile_n, sorted_dir)
            done = (p + 1) / n_pass
            el = time.time() - t_start
            log.info(f"Sort pass {p + 1}/{n_pass} written; {el / 60:.1f} min elapsed, peak RSS {_rss_gb():.2f} GB.")
            if done < 1:
                log.info(f"About {el / done * (1 - done) / 60:.0f} min left in the sort.")

    # ---- per-ratchet tables and constants
    planned = planned_nights(run_dir) if prov else np.unique(cz["night"])
    labels = sorted({dp.half_year(n) for n in planned})
    period = np.array([labels.index(dp.half_year(n)) for n in cz["night"]], np.int16)
    moon = moon_vectors(cz["mjd_mid"])
    n_slots = np.rint(cz["n_epochs"] * cz["exptime_s"] / 60.0).astype(np.int64)
    rat = {k: cz[k] for k in RATCHET_KEYS}
    rat.update(rotation=cz["rotation"], rotation_mid=cz["rotation_mid"], moon_altaz=moon, period=period)
    quant = log_quantisation(float(cz["noise_min"].min()), float(cz["noise_max"].max()))
    if quant["ln_step"] / 2 > TOLERANCE["sigma_bkg_e"]:
        raise RuntimeError(f"sigma_bkg_e spans too wide a range for 16 bits: step {quant['ln_step']:.2e}.")
    tp_table = throughput_table(bands)
    tb = cz["tel_band"]
    band_of_tel = np.full(int(geom.tel_id.max()) + 1, -1, np.int8)
    band_of_tel[tb // 256] = tb % 256
    zps = [
        (manifest["band_zeropoints"][b]["zp_photons_per_sec"], manifest["band_zeropoints"][b]["band_qe"]) for b in bands
    ]
    from .. import c

    state = dict(
        geom=geom.arrays(),
        index_dir=index_dir,
        build_dir=build_dir,
        sorted_dir=sorted_dir,
        ratchet=dict(rat, n_slots=n_slots),
        tp_table=tp_table,
        quant=quant,
        band_of_tel=band_of_tel,
        n_bands=len(bands),
        n_periods=len(labels),
        zeropoints=zps,
        base_cadence_s=float(c.survey.base_cadence_s),
    )

    # ---- stage 2: compact shards, depth maps, checks
    t2 = time.time()
    summaries = [None] * n_shards
    with ctx.Pool(workers, initializer=_init_worker, initargs=(state,)) as pool:
        for i, s in enumerate(pool.imap_unordered(_compact, range(n_shards))):
            summaries[int(s["shard"])] = s
            if (i + 1) % 10 == 0 or i + 1 == n_shards:
                el = time.time() - t2
                log.info(
                    f"Compacted {i + 1}/{n_shards} shards, {el / 60:.1f} min, ETA {el / (i + 1) * (n_shards - i - 1) / 60:.0f} min."
                )
    worst = {k: max(float(s[f"worst_{k}"]) for s in summaries) for k in TOLERANCE}
    n_cov_bad = int(sum(int(s["n_cov_bad"]) for s in summaries))
    log.info(f"Recomputed columns against the run: n_covered differs in {n_cov_bad} rows; largest differences {worst}.")
    over = {k: v for k, v in worst.items() if v > TOLERANCE[k]}
    if over or n_cov_bad:
        raise RuntimeError(
            f"Recomputed columns differ from the run beyond tolerance: {over}, n_covered in {n_cov_bad} rows. "
            "The configuration (layout, site, throughputs) is not the run's."
        )

    # ---- assemble the store
    tiles = {
        k: np.concatenate([s[f"tile_{k}"] for s in summaries]) for k in ("tanseg_id", "row_group", "offset", "n_rows")
    }
    tiles["shard"] = np.concatenate([np.full(len(s["tile_tanseg_id"]), int(s["shard"])) for s in summaries])
    if np.any(tiles["tanseg_id"] != tile_tid) or np.any(tiles["n_rows"] != tile_n):
        raise RuntimeError("The compact shards do not hold the planned tiles.")
    _write_atomic_npz(
        os.path.join(index_dir, "tiles.npz"),
        compressed=True,
        tanseg_id=tiles["tanseg_id"].astype(np.int32),
        shard=tiles["shard"].astype(np.int32),
        row_group=tiles["row_group"].astype(np.int32),
        offset=tiles["offset"].astype(np.int32),
        n_rows=tiles["n_rows"].astype(np.int64),
    )
    _write_atomic_npz(os.path.join(index_dir, "ratchets.npz"), compressed=True, **rat)
    _write_atomic_npz(os.path.join(index_dir, "layout.npz"), **geom.arrays(), band_of_tel=band_of_tel)
    depth_groups = _write_depth(os.path.join(index_dir, "depth.parquet"), summaries, bands, labels)
    _write_nights(run_dir, os.path.join(index_dir, "nights.parquet"))

    from ..provenance import git_state

    span_path = os.path.join(run_dir, "span_summary.json")
    span = json.load(open(span_path)) if os.path.isfile(span_path) else {}
    git = git_state()
    size = sum(os.path.getsize(p) for p in glob.glob(os.path.join(index_dir, "pairs", "*.parquet")))
    kept = np.unique(cz["night"])
    index = {
        "format_version": FORMAT_VERSION,
        "format": (
            "parquet shards; a row group holds one tile or a run of small consecutive tiles (tiles.npz gives the "
            "shard, row group, offset and row count of each tile); rows in ratchet order within a tile"
        ),
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "source_run": run_dir,
        "source_run_git_hash": prov.get("git_hash"),
        "argus_sim_commit": git["git_hash"],
        "argus_sim": git,
        "bands": bands,
        "zeropoints": manifest["band_zeropoints"],
        "plan": {
            "start": prov.get("start"),
            "end": prov.get("end"),
            "n_nights": len(planned),
            "n_kept_nights": len(kept),
            "nights": [int(n) for n in planned],
            "kept_nights": [int(n) for n in kept],
            "night_label": "integer MJD of the local noon that starts the night",
        },
        "periods": labels,
        "settings": _settings(prov, span),
        "depth_convention": manifest.get("depth_convention"),
        "span": span.get("span"),
        "quantisation": {"sigma_bkg_e": quant},
        "recomputed": {
            "columns": ["airmass", "throughput", "field_angle", "n_covered"],
            "airmass": "Pickering (2002) at the tile centre's altitude at mid-ratchet (rotation_mid)",
            "throughput": tp_table["definition"],
            "field_angle": "tile centre (full pairs) or centroid of the covered minipixes (partial pairs)",
            "n_covered": "4096 for full pairs, covered minipixes for partial pairs",
            "tolerance": TOLERANCE,
            "largest_difference": worst,
            "n_covered_differing_rows": n_cov_bad,
        },
        "throughput_table": tp_table,
        "layout": _layout_info(layout),
        "moon_mask": {
            "min_moon_sep_deg": float(c.survey.min_moon_sep),
            "moon_altaz": "Moon unit vector (north, east, up) at mid-ratchet, in ratchets.npz",
        },
        "depth": {
            "file": "depth.parquet",
            "row_groups": depth_groups,
            "limmag": f"{dp.NSIGMA:g}-sigma depth, median over the tile's owned minipixes covered in the period",
            "counts": "n_slots_60, n_slots_1 and n_frames sum over the tile's (tile, OTA) pairs",
        },
        "n_ratchets": len(files),
        "complete_run": limit_ratchets is None and tanseg_range is None,
        "limit_ratchets": limit_ratchets,
        "tanseg_range": None if tanseg_range is None else [int(v) for v in tanseg_range],
        "n_pair_rows": int(tile_n.sum()),
        "n_partial_rows": int(cz["n_partial"].sum()),
        "n_tiles": int(len(tile_tid)),
        "n_shards": int(n_shards),
        "pairs_bytes": int(size),
        "mask_check": {
            "every_nth_ratchet": check_every,
            "bits_checked": int(cz["check_bits"]),
            "bits_differing": int(cz["check_bad"]),
        },
        "slot_convention": "slot k starts at mjd_first - 30 s + k * 60 s (mjd_first is the first exposure's mid-time)",
        "build_seconds": round(time.time() - t_start, 1),
        "build_workers": workers,
        "build_max_memory_gb": max_memory_gb,
        "build_sorted_from": sorted_from and sorted_dir,
    }
    _write_json(os.path.join(index_dir, "index.json"), index)
    man = write_manifest(index_dir)
    if not keep_sorted and sorted_from is None:
        for p in glob.glob(os.path.join(sorted_dir, "pairs", "shard_*.parquet")):
            os.remove(p)
    log.info(
        f"Visit store complete: {man['total_bytes'] / 1e9:.2f} GB in {len(man['files'])} files "
        f"({size / max(index['n_pair_rows'], 1):.2f} B per pair row), {(time.time() - t_start) / 60:.1f} min."
    )
    return index_dir


def _settings(prov: dict, span: dict) -> dict:
    keep = ("start", "end", "coverage", "ratchet_len", "reset_min", "cycle", "seed", "apply_weather")
    out = {k: v for k, v in span.get("settings", {}).items() if k in keep}
    out.update({k: prov[k] for k in ("ring_layout", "band_assignment", "tiling") if k in prov})
    return out


def _layout_info(name: str) -> dict:
    from ..ring_layout import _LAYOUT_DIR, LAYOUTS

    spec = LAYOUTS.get(name)
    if spec is None:
        return {"name": name}

    def sha(fn):
        return hashlib.sha256(_LAYOUT_DIR.joinpath(fn).read_bytes()).hexdigest()

    out = {
        "name": name,
        "file": spec.filename,
        "sha256": sha(spec.filename),
        "az_offset_deg": spec.az_offset_deg,
        "fov_deg": list(spec.fov_deg) if spec.fov_deg else None,
    }
    if spec.band_table:
        out.update(band_table=spec.band_table, band_table_sha256=sha(spec.band_table))
    return out


def _write_depth(path: str, summaries: list, bands: list[str], labels: list[str]) -> dict:
    """Write the depth maps, one row group per (band, period), and return {"band/period": row group}."""
    cols = {k: np.concatenate([s[f"depth_{k}"] for s in summaries]) for k in dp.DEPTH_COLUMNS}
    periods = [*labels, "all"]
    ra, dec = sh.tile_center(cols["tanseg_id"])
    schema = pa.schema(
        [
            ("band", pa.string()),
            ("period", pa.string()),
            ("tanseg_id", pa.int32()),
            ("ra", pa.float64()),
            ("dec", pa.float64()),
            ("limmag", pa.float32()),
            ("covered_fraction", pa.float32()),
            ("n_nights", pa.int32()),
            ("n_slots_60", pa.int64()),
            ("n_slots_1", pa.int64()),
            ("n_frames", pa.int64()),
        ]
    )
    groups = {}
    tmp = path + ".tmp"
    with pq.ParquetWriter(tmp, schema, compression="zstd", compression_level=6) as w:
        for b, band in enumerate(bands):
            for p, label in enumerate(periods):
                sel = np.flatnonzero((cols["band"] == b) & (cols["period"] == p))
                sel = sel[np.argsort(cols["tanseg_id"][sel], kind="stable")]
                data = {
                    "band": np.full(len(sel), band),
                    "period": np.full(len(sel), label),
                    "tanseg_id": cols["tanseg_id"][sel],
                    "ra": np.asarray(ra)[sel],
                    "dec": np.asarray(dec)[sel],
                    **{k: cols[k][sel] for k in dp.DEPTH_COLUMNS[3:]},
                }
                w.write_table(
                    pa.Table.from_arrays([pa.array(data[f.name], f.type) for f in schema], schema=schema),
                    row_group_size=max(len(sel), 1),
                )
                groups[f"{band}/{label}"] = len(groups)
    os.replace(tmp, path)
    return groups


def _write_nights(run_dir: str, path: str) -> None:
    """Write the run's night_summary.csv, with the columns of night_record_stats.csv joined on night."""
    import pandas as pd

    src = os.path.join(run_dir, "night_summary.csv")
    if not os.path.isfile(src):
        raise FileNotFoundError(f"{src} is missing.")
    df = pd.read_csv(src)
    stats_path = os.path.join(run_dir, "night_record_stats.csv")
    if os.path.isfile(stats_path):
        st = pd.read_csv(stats_path)
        st = st[["night", *[c for c in st.columns if c not in df.columns]]]
        df = df.merge(st, on="night", how="left")
    tmp = path + ".tmp"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd")
    os.replace(tmp, path)


def write_manifest(index_dir: str) -> dict:
    """Write ``manifest.json``: every file of the store (``build/`` excluded) with its size and sha256."""
    files = []
    for root, dirs, names in os.walk(index_dir):
        dirs[:] = sorted(d for d in dirs if not (root == index_dir and d == "build"))
        for n in sorted(names):
            p = os.path.join(root, n)
            rel = os.path.relpath(p, index_dir)
            if rel == "manifest.json" or n.endswith(".tmp"):
                continue
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    h.update(chunk)
            files.append({"path": rel, "bytes": os.path.getsize(p), "sha256": h.hexdigest()})
    man = {"format_version": FORMAT_VERSION, "total_bytes": sum(f["bytes"] for f in files), "files": files}
    _write_json(os.path.join(index_dir, "manifest.json"), man)
    return man


def moon_vectors(mjd_mid: np.ndarray) -> np.ndarray:
    """Return the Moon's topocentric unit vector (north, east, up) at each mid-ratchet time, as the survey's mask."""
    import astropy.coordinates as crds
    import astropy.time as atime
    import astropy.units as u

    from .. import c

    t = atime.Time(np.asarray(mjd_mid, np.float64), format="mjd", scale="utc")
    site = crds.EarthLocation(
        lat=c.observatory.latitude * u.deg, lon=c.observatory.longitude * u.deg, height=c.observatory.altitude * u.m
    )
    moon = crds.get_body("moon", t).transform_to(crds.AltAz(location=site, obstime=t))
    alt, az = moon.alt.rad, moon.az.rad
    return np.stack([np.cos(alt) * np.cos(az), np.cos(alt) * np.sin(az), np.sin(alt)], axis=-1)


def planned_nights(run_dir: str) -> np.ndarray:
    """Return the night labels of the survey plan, including nights the weather cull removed.

    The run's ``provenance.json`` gives the start and end. A night counts when the schedule has at least one dark
    exposure in it, as ``Survey.run`` builds the plan. Night labels follow the run: the integer part of
    mjd + longitude / 360 - 0.5 at the first dark exposure of a ratchet.
    """
    import astropy.coordinates as crds
    import astropy.time as atime
    import astropy.units as u

    from .. import c
    from ..coverage_exact import Schedule

    path = os.path.join(run_dir, "provenance.json")
    with open(path) as f:
        prov = json.load(f)
    t0, t1 = atime.Time(prov["start"], scale="utc"), atime.Time(prov["end"], scale="utc")
    lat, lon = c.observatory.latitude, c.observatory.longitude
    schedule = Schedule(
        exptime_s=c.survey.base_cadence_s,
        track_min=float(prov.get("ratchet_len", c.survey.ratchet_len))
        - float(prov.get("reset_min", c.survey.reset_min)),
        reset_min=float(prov.get("reset_min", c.survey.reset_min)),
        sun_alt_max_deg=c.survey.min_sun_alt,
    )
    site = crds.EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=c.observatory.altitude * u.m)

    def dark_nights(a, b):
        ex = schedule.exposures(a, b, site)
        return np.unique(np.floor(ex["time"].mjd + lon / 360.0 - 0.5).astype(np.int64))

    # Every night between the first and last has dark time when |latitude| < 48 deg, so the schedule is only
    # evaluated over the first and last two days.
    edge = 2.0 * u.day
    if abs(lat) >= 48.0 or (t1 - t0) <= 2 * edge:
        return dark_nights(t0, t1)
    head, tail = dark_nights(t0, t0 + edge), dark_nights(t1 - edge, t1)
    return np.union1d(np.union1d(head, np.arange(head.max() + 1, tail.min())), tail)


def _run_census(pool, files: list[str], universe: np.ndarray, check_every: int) -> dict:
    t0 = time.time()
    n = len(files)
    scal = {k: [] for k in RATCHET_KEYS}
    rot = np.empty((n, 3, 3))
    rot_mid = np.empty((n, 3, 3))
    noise_min, noise_max = np.empty(n), np.empty(n)
    tile_counts = np.zeros(len(universe), np.int64)
    n_partial = np.empty(n, np.int64)
    check_bits = check_bad = 0
    tel_band = set()
    check = [(i % check_every == 0) or (i == n - 1) for i in range(n)]
    for i, (s, R, Rm, u, cnt, npart, chk, tb, nr) in enumerate(pool.imap(_census, zip(files, check), chunksize=16)):
        for k in RATCHET_KEYS:
            scal[k].append(s[k])
        rot[i], rot_mid[i] = R, Rm
        noise_min[i], noise_max[i] = nr
        pos = np.searchsorted(universe, u)
        if np.any(universe[np.minimum(pos, len(universe) - 1)] != u):
            raise ValueError(f"{files[i]} has tanseg ids outside the all-sky tessellation.")
        tile_counts[pos] += cnt
        n_partial[i] = npart
        check_bits += chk[0]
        check_bad += chk[1]
        tel_band.update(tb.tolist())
        if (i + 1) % 2000 == 0 or i + 1 == n:
            el = time.time() - t0
            log.info(f"Census {i + 1}/{n} ratchets, {el:.0f} s, ETA {el / (i + 1) * (n - i - 1):.0f} s.")
    out = {k: np.asarray(v) for k, v in scal.items()}
    if np.any(np.diff(out["ratchet"]) <= 0):
        raise ValueError("Ratchet numbers do not increase with the file order.")
    tel_band = np.array(sorted(tel_band), np.int64)
    if len(np.unique(tel_band // 256)) != len(tel_band):
        raise ValueError("An OTA appears in more than one band.")
    out.update(
        rotation=rot,
        rotation_mid=rot_mid,
        noise_min=noise_min,
        noise_max=noise_max,
        tile_counts=tile_counts,
        n_partial=n_partial,
        check_bits=np.int64(check_bits),
        check_bad=np.int64(check_bad),
        tel_band=tel_band,
    )
    return out


def _run_pass(pool, files, p, n_pass, shards, plan, tile_tid, tile_n, sorted_dir) -> None:
    t0 = time.time()
    in_pass = np.isin(plan["shard"], shards)
    tids = tile_tid[in_pass]
    counts = tile_n[in_pass]
    n_rows = int(counts.sum())
    lo, hi = int(tids[0]), int(tids[-1])
    log.info(f"Sort pass {p + 1}/{n_pass}: tanseg {lo}..{hi}, {len(tids):,} tiles, {n_rows:,} rows.")
    cols = {f.name: np.empty(n_rows, f.type.to_pandas_dtype()) for f in SORTED_SCHEMA}
    # Each row goes straight to its tile's next free slot, so rows within a tile stay in file (ratchet) order and
    # the pass needs no argsort or gathered copy.
    cursor = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    end = cursor + counts
    for i, res in enumerate(pool.imap(_extract, ((f, lo, hi) for f in files), chunksize=16)):
        if res is not None:
            ti = np.searchsorted(tids, res["tanseg_id"])
            if np.any(ti >= len(tids)) or np.any(tids[np.minimum(ti, len(tids) - 1)] != res["tanseg_id"]):
                raise RuntimeError(f"Pass {p + 1}: {files[i]} has rows in tiles the census did not count.")
            order = np.argsort(ti, kind="stable")
            ts = ti[order]
            first = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]])
            size = np.diff(np.r_[first, len(ts)])
            dest = cursor[ts] + np.arange(len(ts)) - np.repeat(first, size)
            if np.any(dest >= end[ts]):
                raise RuntimeError(
                    f"Pass {p + 1} read more rows than the census counted. Did the run change since the census?"
                )
            for k, v in res.items():
                cols[k][dest] = v[order]
            cursor[ts[first]] += size
        if (i + 1) % 10000 == 0:
            log.info(f"Sort pass {p + 1}/{n_pass}: read {i + 1}/{len(files)} ratchets, {time.time() - t0:.0f} s.")
    if np.any(cursor != end):
        raise RuntimeError(f"Pass {p + 1} read {int((cursor - end + counts).sum())} rows; the census counted {n_rows}.")
    t1 = time.time()
    if np.any(cols["tanseg_id"][np.cumsum(counts) - 1] != tids):
        raise RuntimeError(f"Pass {p + 1}: the sorted rows do not match the planned tiles.")
    t2 = time.time()
    bounds = np.concatenate([[0], np.cumsum(counts)])
    shard_of = plan["shard"][in_pass]
    for s in shards:
        k = np.flatnonzero(shard_of == s)
        a, b = bounds[k[0]], bounds[k[-1] + 1]
        _write_sorted_shard(
            shard_path(sorted_dir, s), {n: v[a:b] for n, v in cols.items()}, bounds[k[0] : k[-1] + 2] - a
        )
    log.info(
        f"Sort pass {p + 1}/{n_pass}: read and place {t1 - t0:.0f} s, write {time.time() - t2:.0f} s; "
        f"peak RSS {_rss_gb():.2f} GB."
    )
